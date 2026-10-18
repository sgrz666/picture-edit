# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) Facebook, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from src.flux.pipeline_tools import tokenize_t5_prompt

def unpad_input_ids(input_ids, attention_mask):
    return [input_ids[i][attention_mask[i].bool()][:-1] for i in range(input_ids.shape[0])]

def get_word_index(pipe, prompt, input_ids, word, word_count=1, max_length=512, verbose=True, reverse=False):
    word_inputs = tokenize_t5_prompt(pipe, word, max_length)
    word_ids = unpad_input_ids(word_inputs.input_ids, word_inputs.attention_mask)[0]
    if word_ids[0] == 3:
        word_ids = word_ids[1:] # remove prefix space

    if verbose:
        print(f"Trying to find {word} {word_ids.tolist()} in {input_ids.tolist()} where")
        print([(i, pipe.tokenizer_2.decode(input_ids[i])) for i in range(input_ids.shape[0])])
    
    count = 0
    if reverse:
        for i in range(input_ids.shape[0] - word_ids.shape[0],-1,-1):
            if torch.equal(input_ids[i:i+word_ids.shape[0]], word_ids):
                count += 1
                if count == word_count:
                    if verbose:
                        reconstructed_word = pipe.tokenizer_2.decode(input_ids[i:i+word_ids.shape[0]])
                        assert reconstructed_word == word
                        print(f"[Reverse] Found index {i} to {i+word_ids.shape[0]} for '{word}' in prompt '{prompt}'")
                        print("Reconstructed word", reconstructed_word)
                    return i, i + word_ids.shape[0]
    else:
        for i in range(input_ids.shape[0] - word_ids.shape[0] + 1):
            if torch.equal(input_ids[i:i+word_ids.shape[0]], word_ids):
                count += 1
                if count == word_count:
                    if verbose:
                        reconstructed_word = pipe.tokenizer_2.decode(input_ids[i:i+word_ids.shape[0]])
                        assert reconstructed_word == word
                        print(f"Found index {i} to {i+word_ids.shape[0]} for '{word}' in prompt '{prompt}'")
                        print("Reconstructed word", reconstructed_word)
                    return i, i + word_ids.shape[0]
    print(f"[Error] Could not find '{word}' in prompt '{prompt}' with word_count {word_count}")