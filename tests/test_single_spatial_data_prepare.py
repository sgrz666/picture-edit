"""CPU contracts for the private, resumable Bilibili preparation process."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'experiments/bilibili-single-spatial-training/scripts'
sys.path.insert(0, str(SCRIPTS))
import data_prepare as data
import npz_codec as codec
import prepare_dataset as assembly


def test_matching_historical_pair_integrity_failure_stops_instead_of_rebuilding(tmp_path,monkeypatch):
    from types import SimpleNamespace
    old=tmp_path/'old/native_sapiens_v1/pairs/pair/version';old.mkdir(parents=True)
    (old/'ready.json').write_text(json.dumps({'row':{'references':{'sha256':'same-source'}}}))
    args=SimpleNamespace(old_root=tmp_path/'old')
    source={'artifacts':{'references':{'sha256':'same-source'}}}
    def corrupt(*args):raise ValueError('Artifact SHA differs: historical cache')
    monkeypatch.setattr(assembly,'read_reused_pair',corrupt)
    with pytest.raises(ValueError,match='SHA'):
        assembly.reuse_historical_pair(args,{'pair_id':'pair'},source,{}, {},'same-policy')


class CodecTests(unittest.TestCase):
    def test_level3_is_standard_npz_without_codec_member_or_changed_values(self):
        arrays = {'float': np.array([0., -0., np.nan, np.inf], np.float32),
                  'fortran': np.asfortranarray(np.arange(60, dtype=np.int64).reshape(10, 6)),
                  'bool': np.array([True, False]), 'text': np.array('中文 metadata')}
        output = io.BytesIO()
        codec.savez_compressed(output, np.arange(3, dtype=np.uint16), **arrays)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {'arr_0.npy', *(k+'.npy' for k in arrays)})
            self.assertTrue(all(i.compress_type == zipfile.ZIP_DEFLATED for i in archive.infolist()))
        output.seek(0)
        with np.load(output, allow_pickle=False) as actual:
            for name, expected in {'arr_0': np.arange(3, dtype=np.uint16), **arrays}.items():
                self.assertEqual(actual[name].dtype, expected.dtype)
                self.assertEqual(actual[name].shape, expected.shape)
                self.assertEqual(actual[name].tobytes(order='A'), expected.tobytes(order='A'))

    def test_context_restores_numpy_after_error_and_never_adds_compresslevel_array(self):
        original = np.savez_compressed
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with codec.private_numpy_codec():
                self.assertIs(np.savez_compressed, codec.savez_compressed)
                raise RuntimeError('intentional')
        self.assertIs(np.savez_compressed, original)


class PrepareTests(unittest.TestCase):
    def test_memo_rehashes_when_same_size_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data'; path.write_bytes(b'old')
            memo = data.StatHashMemo()
            first = memo(path)
            self.assertEqual(memo(path), first)
            self.assertEqual(memo.hash_reads, 1)
            path.write_bytes(b'new')
            # NTFS timestamp updates can coalesce immediate same-size writes.
            stamp=path.stat();os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1_000_000))
            self.assertNotEqual(memo(path), first)
            self.assertEqual(memo.hash_reads, 2)
            with self.assertRaisesRegex(ValueError, 'SHA'):
                memo.verify({'path':str(path), 'sha256':first})

    def test_immutable_publish_accepts_resume_and_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prepared.json'
            first = data.publish(path, {'a': 1})
            self.assertEqual(data.publish(path, {'a':1}), first)
            with self.assertRaisesRegex(ValueError, 'differs'):
                data.publish(path, {'a':2})
            self.assertEqual(json.loads(path.read_text()), {'a':1})

    def test_interrupted_publication_preserves_complete_pending_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'prepared.json'
            with mock.patch.object(data.os,'link',side_effect=OSError('simulated interruption')):
                with self.assertRaisesRegex(OSError,'interruption'):
                    data.publish(target,{'a':1})
            self.assertFalse(target.exists())
            pending=list(target.parent.glob('*.pending'))
            self.assertEqual(len(pending),1)
            self.assertEqual(json.loads(pending[0].read_text()),{'a':1})
            data.publish(target,{'a':1})
            self.assertEqual(json.loads(target.read_text()),{'a':1})

    def test_completed_preparation_rechecks_manifest_dependency_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            payload=root/'body.npz';payload.write_bytes(b'old')
            manifest=data.publish(root/'manifest.jsonl',[{'body':data.ref(payload)}],True)
            prepared=root/'prepared.json'
            data.publish(prepared,{'schema':'bilibili_single_prepared_v1','source':manifest})
            data.verify_prepared(prepared)
            payload.write_bytes(b'new')
            stamp=payload.stat();os.utime(payload,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(ValueError,'SHA'):
                data.verify_prepared(prepared)

    def test_fine_hand_validity_remains_exactly_original(self):
        frame = {'frame_id':'f', 'clip_id':'c', 'split':'train', 'rgb_path':'/x/rgb.png',
                 'rgb_sha256':'a'*64, 'loss_mask':{'hand_a':False}, 'hand_raw_pose_usable':{'hand_a':False}}
        record = {'frame_id':'f', 'clip_id':'c', 'split':'train', 'group_id':'c',
                  'rgb':{'path':'/x/rgb.png','sha256':'a'*64}, 'loss_mask':{'hand_a':False},
                  'geometry_method':'sapiens_v1', 'canvas':{'grid':'exact'},
                  'source_features_complete':True, 'feature_protocol':'FP32_same_models_and_preprocessing_batched_16'}
        data.check_frame_binding(frame, record, {'grid':'exact'})
        record['loss_mask']={'hand_a':True}
        with self.assertRaisesRegex(ValueError, 'loss_mask'):
            data.check_frame_binding(frame, record, {'grid':'exact'})

    def test_source_feature_identity_reuses_original_producer_not_new_run_root(self):
        models={'device':'cuda','model':'original-frozen-model'}
        producer={'path':'/old/source/appearance.py','sha256':'a'*64}
        record={'physical_identity':'physical'}
        record['identity']=data.digest({'physical_frame':'physical','models':models,
                                       'feature_producer':producer,'batch_limit':16})
        assembly.check_feature_identity(record,models,producer)
        record['physical_identity']='different'
        with self.assertRaisesRegex(ValueError,'feature identity'):
            assembly.check_feature_identity(record,models,producer)

    def test_pixel_identity_decodes_rgb_instead_of_trusting_file_encoding(self):
        from PIL import Image
        image=Image.fromarray(np.arange(120,dtype=np.uint8).reshape(4,10,3))
        first=io.BytesIO();second=io.BytesIO()
        image.save(first,format='PNG',compress_level=0)
        image.save(second,format='PNG',compress_level=9)
        first.seek(0);second.seek(0)
        with Image.open(first) as a,Image.open(second) as b:
            self.assertEqual(assembly.decoded_rgb_sha(a),assembly.decoded_rgb_sha(b))

    def test_spatial_config_requires_exact_full_annotations_body_and_canvas(self):
        ref={'path':'/frames.jsonl','sha256':'a'*64}; body={'path':'/body/index.json','sha256':'b'*64}
        config={'frames':ref,'body_cache_index':body,'hand_policy':'visible_region_v2',
                'canvas_long_side':512,'geometry_method':'sapiens_v1','selected_frame_ids':['f1','f2']}
        data.check_spatial_config(config,ref,body,['f1','f2'])
        config['selected_frame_ids']=['f1']
        with self.assertRaisesRegex(ValueError, 'coverage'):
            data.check_spatial_config(config,ref,body,['f1','f2'])

    def test_spatial_resume_command_uses_same_output_cpu_and_original_parameters(self):
        from types import SimpleNamespace
        args=SimpleNamespace(repo_root=Path('/repo'),data_root=Path('/data'),spatial_root=Path('/spatial'),
                             body_index=Path('/body/index.json'))
        config={'sam_config':'configs/a.yaml','seed':42,'geometry_method':'sapiens_v1',
                'device':'cpu','torch_threads':4,'canvas_long_side':512,'hand_policy':'visible_region_v2'}
        command=data.spatial_resume_command(args,config)
        self.assertIn('--resume',command)
        self.assertEqual(command[command.index('--output-root')+1],str(args.spatial_root))
        self.assertEqual(command[command.index('--device')+1],'cpu')
        self.assertEqual(command[command.index('--limit')+1],'0')

    def test_prepared_separates_holdout_and_binds_all_source_spatial_refs(self):
        manifests={s:{'manifest':{'path':'/'+s,'sha256':'a'*64},'groups':{'path':'/'+s+'.groups','sha256':'b'*64}}
                   for s in ('train','val','test')}
        spatial={'path':'/spatial/index.json','sha256':'c'*64}
        result=data.make_prepared(manifests,spatial,{'path':'/init','sha256':'d'*64},
            {'path':'/model.json','sha256':'e'*64},Path('/run'),data.EXPECTED_COUNTS,{})
        self.assertEqual(result['schema'],'bilibili_single_prepared_v1')
        self.assertEqual([s['source_id'] for s in result['sources']],['bilibili_train','bilibili_val'])
        self.assertEqual([s['role'] for s in result['sources']],['train','validation'])
        self.assertEqual(result['test']['source_id'],'bilibili_test')
        self.assertTrue(all(s['spatial_weights']==spatial for s in result['sources']+[result['test']]))
        self.assertEqual(result['counts']['pairs'],12156)

    def test_reused_pair_requires_float_scene_exact_source_target_and_fine_flags(self):
        from types import SimpleNamespace
        source={'rgb':{'sha256':'source'},'canvas':{'grid':'s'},'artifacts':{'asset':{'sha256':'sa'}},
                'original_geometry':{'parameters':'source_v1'}}
        target={'rgb':{'sha256':'target'},'canvas':{'grid':'t'},'artifacts':{'asset':{'sha256':'ta'}},
                'original_geometry':{'parameters':'target_v1'}}
        pair={'pair_id':'p','split':'train','clip_id':'c','source_loss_mask':{'hand_a':False},
              'target_loss_mask':{'hand_a':False}}
        frame={'loss_mask':{'hand_a':False},'hand_raw_pose_usable':{'hand_a':False}}
        metadata={'source_asset_sha256':'sa','driver_record_sha256':'ta','source_rgb_sha256':'source',
                  'driver_rgb_sha256':'target','loss_mask':frame['loss_mask'],
                  'hand_raw_pose_usable':frame['hand_raw_pose_usable'],
                  'local_scene_depth':{'policy':'float_exact','source_asset_sha256':'ta','target_canvas_id':'tc'}}
        metadata['target_canvas_id']='tc'
        pairmeta={'pair_id':'p','source_rgb_sha256':'source','target_rgb_sha256':'target',
                  'source_asset_sha256':'sa','driver_asset_sha256':'ta','geometry_method':'sapiens_v1',
                  'source_loss_mask':pair['source_loss_mask'],'target_loss_mask':pair['target_loss_mask'],
                  'selected_source_geometry':source['original_geometry'],'selected_target_geometry':target['original_geometry']}
        sample={'pair_metadata':pairmeta,'source_canvas':source['canvas'],'target_canvas':target['canvas'],
                'local_condition_v4':SimpleNamespace(metadata=[metadata])}
        assembly.check_pair_binding(pair,source,target,frame,sample,'float_exact')
        metadata['local_scene_depth']['policy']='png_quantized'
        with self.assertRaisesRegex(ValueError,'float scene'):
            assembly.check_pair_binding(pair,source,target,frame,sample,'float_exact')


if __name__ == '__main__':
    unittest.main()
