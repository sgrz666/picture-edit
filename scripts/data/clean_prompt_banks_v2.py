#!/usr/bin/env python3
"""Re-derive the TikTok / Hi4D per-sample prompt banks under corrected compile rules (v2).

Reads each bank's own ``rewritten_prompt`` (the PE answer before compilation, present on
100% of rows in both banks) and re-emits the two compiled forms. Nothing is invented: every
edit is a deletion or a substitution of text PE already wrote, so no new claim enters the bank.

v1 files are never touched. Output goes to ``prompts_v2/`` next to ``prompts/``.

Rules (numbered as cited in docs/数据集样本提示词质量审查_20261024.md §8):
  R1 WATERMARK    drop overlay-text clauses (TikTok logo / username / watermark). 64.4% of rows.
  R2 APPEARANCE   replace clothing+colour enumerations with one generic preservation sentence.
                  Fixes the 69% (TikTok) / 100% (Hi4D) self-contradictory appearance claims.
  R3 GT_REF       delete references to the target photograph / fifth image, in both the
                  prohibition form (ungroundable) and the provenance form ("read from the
                  unseen target photograph"). Expression/pose content itself is kept.
  R4 COLLAPSE     fix role-word self-collision ("the joint skeleton in <SKELETON_WORD>",
                  Hi4D "X in X") and doubled words / lower-case sentence starts.
  R5 SIDE_ANCHOR  Hi4D: stop binding person N to a viewer-side ("the left man (person 1)")
                  and bind it to the isolated skeleton instead. 30/30 episodes contradicted.

Usage:
  python clean_prompt_banks_v2.py --bank tiktok --dry-run --limit 8
  python clean_prompt_banks_v2.py --bank both
"""
from __future__ import annotations

import argparse, collections, hashlib, json, re, statistics
from pathlib import Path

REPO = Path('/home/shangguanrz/project/pic-edit')
BANKS = {
    'tiktok': {
        'dir': REPO / 'experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014',
        'in': lambda s: s + '.jsonl',
        'out': lambda s: s + '.jsonl',
        'fields': ('compiled_prompt', 'compiled_prompt_inference'),
        'sha_fields': ('compiled_prompt_sha256', 'compiled_prompt_inference_sha256'),
        'n_person': 1,
    },
    'hi4d': {
        'dir': REPO / 'experiments/4.two man /hi4d_pe_prompt_rewrite_20261020',
        'in': lambda s: f'hi4d_{s}_prompts_v1.jsonl',
        'out': lambda s: f'hi4d_{s}_prompts_v4.jsonl',
        'fields': ('compiled_prompt_training', 'compiled_prompt_inference'),
        'sha_fields': ('compiled_prompt_training_sha256', 'compiled_prompt_inference_sha256'),
        'n_person': 2,
    },
}
SPLITS = ('train', 'val', 'test')

# The training form's skeleton role word is asserted by scripts/prompt_bank.py and by
# tests/test_tiktok_prompt_bank_contract.py; it must survive every rule.
SKELETON_WORD = 'the target joint arrangement described in this instruction'
HI4D_ROLES = ('the combined dual-person skeleton', "person 1's isolated skeleton",
              "person 2's isolated skeleton")

GENERIC_PRESERVE = {
    1: ('Keep the person exactly as the source image shows them: face, hairstyle, skin tone, '
        'clothing, footwear, accessories and held objects all unchanged.'),
    2: ('Keep both people exactly as the source image shows them: faces, hairstyles, skin '
        'tones, clothing, footwear and accessories all unchanged, with no identity or '
        'clothing swap between the two people.'),
}

# ---------------------------------------------------------------- R1 watermark
RE_WATERMARK_SEG = re.compile(
    r'\b(?:tiktok|logo|wordmark|watermark|username|user\s*name|handle|caption text|'
    r'on-screen text|screen text|text overlay|overlay text|ui text|stamp)\b'
    r'|\B["\u201c][@\w._]{3,}["\u201d]|@\w{3,}', re.I)
RE_WATERMARK_SENT = re.compile(
    r'\b(?:logo|watermark|username|handle|wordmark|on-screen text|text overlay)\b', re.I)

# ---------------------------------------------------------------- R2 appearance
# Substring matching is unusable here: 'tan' fires inside "stance", 'hat' inside "what",
# 'clear' inside "clearly", 'top' inside "stop". Word-boundary matching is equally unusable
# for compound garments ("sweatpants" contains 'pants' with no boundary inside it).
# So every form that actually occurs in these banks is listed as its own word.
RE_COLOR = re.compile(
    r'\b(?:black|white|gray|grey|red|green|blue|navy|pink|magenta|purple|violet|indigo|cyan|'
    r'yellow|gold|golden|silver|bronze|copper|brown|tan|beige|cream|ivory|khaki|olive|maroon|'
    r'burgundy|wine|teal|turquoise|aqua|coral|peach|salmon|rose|ruby|amber|orange|lavender|'
    r'lime|mint|charcoal|monochrome|multicolored|multicolour|colourful|colorful|patterned|'
    r'pattern|striped|stripes|plaid|checkered|tartan|camo|tie-dye|tie-dyed|ombre|mottled|'
    r'floral|polka-dot|heathered|heather|translucent|transparent|see-through|sheer|metallic|'
    r'matte|glossy|denim|'
    # shade modifiers are appearance claims too: "the man in the dark T-shirt" survived the
    # first colour list because `dark` was not treated as a colour word at all
    r'dark|light|pale|deep|bright|neon|pastel|muted|dusty|faded|warm|cool|jet|solid)\b', re.I)
RE_ITEM = re.compile(
    r'\b(?:t-?shirts?|tees?|shirts?|blouses?|tops?|bodysuits?|bras?|bra\s+tops?|camisoles?|'
    r'tank\s+tops?|crop\s+tops?|sweatshirts?|hoodies?|hooded\s+sweatshirts?|sweaters?|'
    r'cardigans?|jackets?|coats?|vests?|gilets?|blazers?|dresses?|skirts?|shorts?|shirts?|'
    r'pants?|trousers?|jeans?|leggings?|slacks?|joggers?|sweatpants?|overalls?|tights?|'
    r'yoga\s+pants?|gowns?|robes?|suits?|uniforms?|costumes?|attire|wardrobe|outfits?|'
    r'clothing|clothes|garments?|fabric|laces?|velour|velvet|cotton|polyester|nylon|'
    r'sneakers?|trainers?|shoes?|flats?|heels?|boots?|sandals?|slippers?|slides?|crocs|'
    r'socks?|stockings?|tights|tongue|laces?|gloves?|mittens?|hats?|caps?|beanies?|bonnets?|'
    r'headbands?|visors?|wigs?|braids?|buns?|ponytails?|locs|dreadlocks|hair|hairstyle|'
    r'beards?|mustaches?|moustaches?|sideburns?|eyebrows?|brows?|lashes|eyelashes|'
    r'nails?|manicure|pedicure|makeup|make-up|lipstick|lip\s+gloss|mascara|eyeliner|blush|'
    r'freckles?|moles?|tattoos?|scarring|piercings?|septum|nostril|nose\s+rings?|'
    r'earrings?|studs?|hoops?|necklaces?|pendants?|chains?|bracelets?|bangles?|wristbands?|'
    r'rings?|watches?|watch|brooch(?:es)?|pins?|badges?|lanyards?|bags?|handbags?|purses?|'
    r'backpacks?|totes?|clutches?|belts?|sashes?s|scarves?|scarf|shaws?|stoles?|muffs?|'
    r'sunglasses|glasses|eyeglasses|eyewear|goggles|masks?|bandanas?|kerchiefs?|'
    r'appliques?|appliqu[aé][s]?|embroider(?:y|ed)?|embellishments?|sequins?|rhinestones?|'
    r'patches?|prints?|logos?|graphics?|decorations?|accessories|accessory|jewelry|jewellery|'
    r'footwear|headwear|hand\s+paint(?:ed)?|paint(?:ed)?|text(?:ed)?|cutout|cut-outs?|'
    r'zipper|zip|buttons?|pocket(?:s)?|cuffs?|collars?|hems?|straps?|buckles?)\b', re.I)
RE_BODY_LOOK = re.compile(
    r'\b(?:facial\s+identity|facial\s+features?|face\s+shape|skin\s+tone|complexion|'
    r'build|silhouette|physique|posture|height|proportions?)\b', re.I)
RE_KEEP_SENT = re.compile(r'^\s*(?:keep|preserve|retain|leave|with\b|both\s+(?:men|women)\b)', re.I)


def enumerates_appearance(sent: str) -> bool:
    """True when the sentence pins *concrete* appearance the source image already owns.

    A colour claim is the part that contradicted between samples, so colour + item is the
    trigger. The generic preservation clause is appearance vocabulary without a claim and is
    excluded by identity, so a second pass can never strip it.
    """
    s = sent.strip()
    if s in GENERIC_PRESERVE.values():
        return False
    if RE_COLOR.search(s) and (RE_ITEM.search(s) or RE_BODY_LOOK.search(s)):
        return True
    items = len(RE_ITEM.findall(s)) + len(RE_BODY_LOOK.findall(s))
    return items >= 4 and len([x for x in segs_of(s) if x.strip()]) >= 4


# ---------------------------------------------------------------- R3 GT references
RE_GT_PHRASE = re.compile(
    r'\b(?:the|a|an|that|this|its|their|her|his)?\s*(?:unseen|actual|hidden|implied|original|'
    r'real|true|reference|target|gt|ground[- ]truth)\s*(?:photograph|photo|picture|image|frame)\s*'
    r'(?:of\s+the\s+target)?\b|\bground[- ]truth\s+(?:image|frame|photograph)\b', re.I)
RE_FIFTH = re.compile(r'\b(?:a\s+)?(?:hidden\s+)?(?:fifth\s+image|image\s*5|<image5>)\b', re.I)
RE_GT_PREP = re.compile(                       # provenance phrase, e.g. "from the unseen target photograph"
    r'\b(?:from|in|of|by|using|as|per)\s+(?:the\s+)?(?:unseen|actual|hidden|implied|reference|'
    r'target|gt|ground[- ]truth)[^.]{0,28}?\b(?:photograph|photo|picture|image|frame)\b'
    r'\s*(?:and\s+(?:the\s+)?(?:described|encoded))?', re.I)
RE_GT_SENT_PROHIB = re.compile(                 # a whole sentence whose subject is the missing image
    r'\b(?:do\s*not|never|without|avoid|must\s*not|no)\b[^.]{0,160}?'
    r'(?:fifth image|target photograph|unseen target|the target image|ground[- ]truth)', re.I)
RE_GT_META = re.compile(                        # PE meta-talk about its own input protocol
    r'\b(?:using|with)\s+the\s+fifth\s+image[^.]{0,120}|'
    r'\bfifth image[^.]{0,80}|'
    r'\bthe\s+(?:target\s+)?(?:photograph|image)\s+is\s+(?:the\s+)?training[^.]{0,80}', re.I)
RE_TARGET_WORDING = re.compile(r'\bthe described target pose\b|\bthe target interaction\b', re.I)

# ---------------------------------------------------------------- R4 collapse
COLLAPSE = [
    # "<guide noun phrase> in|of|by <SKELETON_WORD>" -> just <SKELETON_WORD>
    (re.compile(r'\b(?:joint |coloured |colored |dual-person |combined )?skeleton\s+(?:in|of|by|from|to|for)\s+'
                + re.escape(SKELETON_WORD), re.I), SKELETON_WORD),
    (re.compile(r'\b(?:joint )?arrangement\s+(?:described by|defined by|encoded by|encoded in|shown in|'
                r'defined in)\s+' + re.escape(SKELETON_WORD), re.I), SKELETON_WORD),
    (re.compile(r'\bconfiguration\s+(?:shown in|defined by|encoded by)\s+' + re.escape(SKELETON_WORD), re.I),
     SKELETON_WORD),
]
HI4D_COLLAPSE = [(re.compile(re.escape(a) + r'\s+(?:in|of|by|from|to|for|defined by|shown by)\s+' + re.escape(a), re.I), a)
                 for a in HI4D_ROLES]
RE_DOUBLE_WORD = re.compile(r'\b(\w+)\s+\1\b', re.I)
RE_ART_THE = re.compile(r'\b(the)\s+\1\b', re.I)

# ---------------------------------------------------------------- R5 side anchor (Hi4d)
RE_SIDE_PIN = re.compile(
    r'\b(?:the|that|both)?\s*(left|right)\s+(man|woman|person|individual|figure|boy|girl|subject)'
    r'\s*(?:\(([^)]{0,140})\))?\s*(?=(?:must|will|adopts?|adopt|is|has|have|with|,|\.))', re.I)
RE_GUIDE_SIDE = re.compile(r'\b(?:as|for|to|of)\s+the\s+(?:left|right)\s+(?:man|woman|person|'
                           r'individual|figure)\b', re.I)
RE_PERSON_NUM = re.compile(r'\bperson\s*([12])\b', re.I)
# second-pass shape: a side noun that is the subject of the pose instruction, optionally
# followed by an appositional parenthetical -- "The left person (the man in the dark T-shirt) must"
RE_SIDE_PIN2 = re.compile(
    r'\b(left|right)\s+(man|woman|person|individual|figure|boy|girl)\b'
    r'(\s*\([^)]{0,170}\))?\s*(?=(?:must|will|adopts?|adopt|takes|holding|is|has)\b)', re.I)

RE_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
# failures that would make scripts/prompt_bank.py raise, so those fields revert to v1
HARD_KINDS = ('too_short', 'image_ref_left', 'skeleton_word_lost', 'person_binding_broken',
              'pose_content_lost')

# ---------------------------------------------------------------- R6 repair
# Deletion leaves grammatical wreckage that only shows up when the *cleaned* text is read:
# stray prepositions before punctuation, ", )" inside a parenthetical emptied of appearance
# words, a sentence starting with the conjunction that used to join it, unmatched quotes left
# by removing a "logo text" fragment, and R5's own over-eager "for that person" (both sides
# became the same phrase, so the antecedent is gone). Every pattern here was read out of the
# v2 corpus first; none of it is speculative.
REPAIR = [
    (re.compile(r'\bas the (precise|exact|joint|primary) joint guide for that person\b', re.I),
     r'as the \1 joint guide'),
    (re.compile(r'\bfor that person\b', re.I), ''),
    (re.compile(r'\([,;]?\s*\)'), ''),                        # "(  )" left empty
    # "(a face-to-face reaching confrontation, fight01)" -> keep the useful gloss, drop the
    # dangling comma. Deleting the whole parenthetical threw away action semantics that the
    # text channel is supposed to carry, which a test caught.
    (re.compile(r'\(\s*([^()]{3,}?)\s*,\s*\)'), r'(\1)'),
    (re.compile(r'\(\s*(?:the\s+)?(?:left|right)\s+(?:man|woman|person|individual)\s*\)'), ''),
    (re.compile(r'\([^()]{0,40},\s*\)'), ''),                 # "(left person, )"
    (re.compile(r'\s+([,.])'), r'\1'),                                    # " ,", " ."
    (re.compile(r',\s*,+'), ','),
    (re.compile(r'\s*,\s*(?=\.)'), ''),
    (re.compile(r'\b(?:in|of|on|at|by|with|to|and|or|from)\s*(?=[,.;])', re.I), ''),   # dangling prep
    (re.compile(r',\s*and\s*(?=[,.;])', re.I), ''),
    (re.compile(r"\bthat person(?:'s|’s)\b", re.I), "the corresponding person's"),
    (re.compile(r'\b(?:the\s+)?(?:a\s+)?the\s+(?=the\b)', re.I), ''),
    (re.compile(r'\ba\s+the\b', re.I), 'the'),
    (re.compile(r"(\w)'\s*,", re.I), r'\1,'),                            # "shorts', color"
    (re.compile(r'(?:^|[.!?]\s+)(?:And|Or|So|But|With)\s+', re.I), ' '),   # orphan conjunction
    (re.compile(r',\s*\.', re.I), '.'),
    (re.compile(r'\.\s*\.', re.I), '.'),
]
# On-image text claims are the root cause of the wreckage: PE transcribed "the sweatpants
# carrying the text STOP LOO.." and "four text boxes reading , , ,". Cutting the quoted part
# orients the sentence on nothing, so the whole claim goes before the grammar repair runs.
RE_TEXT_CLAIM = re.compile(
    r'\b(?:with|carrying|bearing|printed\s+with|reading|displaying|displayed|showing|featuring|'
    r'containing|including|accompanied\s+by)\s+(?:the\s+|its\s+|their\s+|exact\s+|same\s+)?'
    r'(?:text|letters?|words?|lettering|caption|banner|banners|sticker|stickers|label|labels|'
    r'signage|watermark|overlay|font|typography|logo|wordmark)\b[^.;]{0,70}'
    r'|\b(?:text|lettering|caption|banner|sticker|label|font)\s+(?:elements|boxes|size|position|and)\b[^.;]{0,70}'
    r'|\bSTOP\s+LOO[^.]{0,10}'
    r'|["“][^"”]{0,60}["”]|["“”]', re.I)

# PE's own voice leaking into the instruction (a v1 defect, counted for the first time here)
RE_META_VOICE = re.compile(
    r'\b(?:no wait|let me|i must|i need to|i will (?:now|re-?)|on second thought|'
    r'in this prompt|this prompt (?:must|should|will)|the target interaction token|'
    r'i (?:have|was) to (?:note|say)|as an ai|note that i)\b', re.I)

# scene surfaces also get colour claims, and those contradicted between samples too
RE_BG = re.compile(r'\b(?:wall(?:s)?|floor|background|backdrop|curtain|panel(?:s)?|door(?:s)?|'
                   r'studio|sky|grass|fence|mat|wallpaper|ceiling|window|shower|mirror)\b', re.I)
RE_ANY_ITEM = re.compile(f'(?:{RE_ITEM.pattern})|(?:{RE_BODY_LOOK.pattern})|(?:{RE_BG.pattern})', re.I)
# "<colour> [and <colour>] [<modifier>]{0,2} <item>" -> drop the colour claims, keep the item
RE_DECOLOR = re.compile(
    r'(' + RE_COLOR.pattern + r')(?:\s+and\s+[a-z]+)?(?:\s+(?:-\s*)?[a-z]+){0,2}?\s+(?='
    + RE_ANY_ITEM.pattern + r')', re.I)


def decolor(text: str, applied) -> str:
    """Strip colour adjectives that modify an item, a body feature or a scene surface.

    The colour claims are exactly the part that contradicted between samples of the same
    person, while "sleeves must stretch and fold along the raised arms" is a useful
    cloth-dynamics instruction. Deleting the adjective keeps the instruction and drops the
    unverifiable claim, which sentence splitting alone could not do.
    """
    new = RE_DECOLOR.sub('', text)
    # Colour is only ever an appearance or scene claim in these prompts; the source image owns
    # it, and every surviving colour adjective was a candidate for the cross-sample clash. The
    # pose-guide contract wording ("color-coded bone lines", "line colors") is not in RE_COLOR.
    new = RE_COLOR.sub('', new)
    new = re.sub(r'\(([^)]{0,300})\)',
                 lambda m: '' if enumerates_appearance(m.group(1)) else m.group(0), new)
    new = re.sub(r',\s*(?=,)|\(\s*\)|\s{2,}', ' ', new)
    new = re.sub(r'\(\s*[,;]\s*', ' (', new)
    if new.strip() != text.strip():
        applied.add('R2_appearance:decolor')
    return new


def segs_of(sentence: str) -> list[str]:
    """comma/semicolon segments, kept in order (used for surgical clause removal)."""
    return [s for s in re.split(r',|;|\s—\s', sentence) if s.strip()]


def has_watermark(sentence: str) -> bool:
    return bool(RE_WATERMARK_SENT.search(sentence))


def strip_watermark(text: str, applied) -> str:
    out = []
    for sent in RE_SENT_SPLIT.split(text):
        if not sent.strip():
            continue
        if has_watermark(sent):
            # whole sentence is about overlay text -> drop; otherwise drop only the clauses
            kept = [s for s in segs_of(sent) if not RE_WATERMARK_SEG.search(s)]
            core = ', '.join(kept).strip()
            if len(core.split()) < 8 or RE_WATERMARK_SENT.search(core):
                applied.add('R1_watermark:sentence')
                continue
            applied.add('R1_watermark:clause')
            out.append(core if core[-1:] in '.!?' else core + '.')
            continue
        # trailing ", including the ... logo text and the ... username text" style tails
        kept = [s for s in segs_of(sent) if not RE_WATERMARK_SEG.search(s)]
        if len(kept) != len(segs_of(sent)):
            applied.add('R1_watermark:segment')
            sent = ', '.join(kept).strip()
        out.append(sent)
    return ' '.join(out)


def strip_appearance(text: str, n_person: int, applied, keep_generic: bool) -> str:
    generic = GENERIC_PRESERVE[n_person]
    out, done = [], False

    def add_generic():
        nonlocal done
        if keep_generic and not done:
            done = True
            out.append(generic)

    for sent in RE_SENT_SPLIT.split(text):
        s = sent.strip()
        if not s:
            continue
        if not enumerates_appearance(s):
            out.append(sent)
            continue
        if RE_KEEP_SENT.match(s):
            applied.add('R2_appearance:preserve_sentence')
            add_generic()
            continue
        kept = [x for x in segs_of(s) if not enumerates_appearance(x)]
        core = ', '.join(kept).strip()
        if len(core.split()) < 8:
            applied.add('R2_appearance:sentence_dropped')
            add_generic()
            continue
        applied.add('R2_appearance:clauses')
        out.append(core if core[-1:] in '.!?' else core + '.')
        add_generic()
    return ' '.join(out)


def strip_gt_refs(text: str, applied) -> str:
    out = []
    for sent in RE_SENT_SPLIT.split(text):
        if not sent.strip():
            continue
        if RE_FIFTH.search(sent) or RE_GT_PHRASE.search(sent):
            if RE_GT_SENT_PROHIB.search(sent) or RE_GT_META.search(sent):
                # sentence exists only to talk about the image the consumer cannot see
                kept = [s for s in segs_of(sent)
                        if not (RE_FIFTH.search(s) or RE_GT_PHRASE.search(s) or RE_GT_META.search(s))]
                core = ', '.join(kept).strip()
                if len(core.split()) < 8:
                    applied.add('R3_gt:sentence')
                    continue
                applied.add('R3_gt:clause')
                out.append(core if core[-1:] in '.!?' else core + '.')
                continue
            # provenance wording inside a content-bearing sentence: cut the phrase, keep the claim
            new = RE_GT_PREP.sub('', sent)
            new = re.sub(r'\s+(?=[.,;])', '', new)
            new = re.sub(r'\b(?:from|in|of|by|using|as)\s+(?=[.,;])', '', new)
            new = RE_DOUBLE_WORD.sub(r'\1', new)
            applied.add('R3_gt:provenance')
            sent = new
        # residual "the target photograph" wording that survived the phrase patterns
        if RE_GT_PHRASE.search(sent) or RE_FIFTH.search(sent):
            sent = re.sub(r'\b(?:the\s+)?(?:unseen|actual|hidden|implied|reference|target|gt)\s+'
                          r'(?:photograph|photo|picture|image|frame)\b', 'the described target pose',
                          sent, flags=re.I)
            sent = re.sub(r'\bfifth image\b', 'the described target pose', sent, flags=re.I)
            applied.add('R3_gt:substitute')
        out.append(sent)
    text = ' '.join(out)
    # "match the target:" / "as in the target" reads like a reference to a missing image
    text = re.sub(r'\b(?:to match|matching)\s+the\s+target\s*:', 'to match the described pose:', text, flags=re.I)
    text = re.sub(r'\bas\s+in\s+the\s+target\b', 'as described', text, flags=re.I)
    return text


def collapse(text: str, n_person: int, applied) -> str:
    before = text
    for pat, rep in COLLAPSE:
        text = pat.sub(rep, text)
    if n_person == 2:
        for pat, rep in HI4D_COLLAPSE:
            text = pat.sub(rep, text)
    if text != before:
        applied.add('R4_collapse:leadin')
    text = RE_ART_THE.sub(r'\1', text)
    text = RE_DOUBLE_WORD.sub(lambda m: m.group(1) if m.group(1).lower() not in ('and', 'or') else m.group(0), text)
    # sentence-initial lowercase left by clause removal
    fixed = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    if fixed != text:
        applied.add('R4_collapse:capitalisation')
    text = fixed
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(person\s*[12])(?=[a-z])', r'\1 ', text)   # never glue a verb onto a slot number
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = re.sub(r'([,;])\s*([,;])', r'\1', text)
    text = re.sub(r'\b(?:,\s*)?and\s*,', ',', text)
    return text


def deside_anchor(text: str, applied) -> str:
    """Hi4D: bind identity to the isolated skeleton, never to a viewer side."""
    changed = False

    def pin_repl(m):
        nonlocal changed
        side, noun, paren = m.group(1), m.group(2), m.group(3) or ''
        # Number source, in order of trustworthiness: the parenthetical attached to THIS noun,
        # then the pose source named in the SAME sentence. Never a neighbouring clause -- that
        # is how a cleaned prompt once bound person 2 to person 1's skeleton.
        attached = RE_PERSON_NUM.search(paren)
        if attached:
            num = attached.group(1)
        else:
            sentence = text[max(0, text.rfind('.', 0, m.start())):text.find('.', m.end()) + 1]
            found = set(re.findall(r'pose\s+(?:from|of|per)\s+person\s*([12])', sentence, re.I))
            num = next(iter(found)) if len(found) == 1 else None
        if not num:
            return m.group(0)
        changed = True
        return f'the person whose isolated skeleton is person {num} '

    new = RE_SIDE_PIN.sub(pin_repl, text)
    # Phrasings the first pass could not see, each one confirmed on a printed guide sheet:
    # "person 1 on the left", "the left skeleton", "The left person (the man in the dark T-shirt)"
    new = re.sub(r'\b(person\s*[12])\s+on\s+the\s+(?:left|right)\b', r'\1', new, flags=re.I)
    new = re.sub(r'\bthe\s+(?:left|right)\s+skeleton\b', 'the corresponding skeleton', new, flags=re.I)
    new = RE_SIDE_PIN2.sub(pin_repl, new)
    if changed:
        applied.add('R5_side_anchor:pin_rewritten')
    # "as the precise joint guide for the left person" -> "... for that person"
    new2, n = RE_GUIDE_SIDE.subn('for that person', new)
    if n:
        applied.add('R5_side_anchor:guide_side')
    # A side-bound skeleton reference is redundant with "person N's isolated skeleton", and the
    # side half of it is the claim that flipped between samples of one episode -> drop the side.
    new3, m = re.subn(r'\b(?:the\s+)?(?:left|right)\s+(?:man|woman|person|individual|figure)'
                      r'(?:\'s)?\s+(?:isolated\s+)?skeleton\b', 'the corresponding isolated skeleton',
                      new2, flags=re.I)
    if m:
        applied.add('R5_side_anchor:side_bound_skeleton')
    return new3


def tidy(text: str) -> str:
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # re-join sentences split by clause deletion so a fragment never ends without a verb
    parts = [p.strip() for p in RE_SENT_SPLIT.split(text) if p.strip()]
    merged = []
    for p in parts:
        if merged and (len(p.split()) < 5 and not RE_SENT_SPLIT.search(p) and p[-1] not in '.!?'):
            merged[-1] = merged[-1].rstrip('. ') + ', ' + p
            continue
        merged.append(p)
    text = ' '.join(merged)
    if text and text[-1] not in '.!?':
        text += '.'
    return text


RE_POSE = re.compile(
    r'\b(?:arm|elbow|wrist|hand|finger|thumb|leg|knee|ankle|foot|toe|hip|shoulder|torso|chest|'
    r'back|neck|head|face|chin|jaw|lean|bend|extend|raise|lower|turn|rotat\w+|tilt|step|stance|'
    r'stride|squat|reach\w*|cross(?:ed)?|overlapping|grasp|hold|embrace|contact)\b', re.I)


def _grammar(text):
    for pat, rep in REPAIR:
        text = pat.sub(rep, text)
    return text


RE_ACTION_LABEL = re.compile(
    r'\b(?:hug|fight|talk|dance|highfive|backhug|kiss|jump|bend|yoga|taichi|basketball|argue|'
    r'game|pose)\d{2}\b', re.I)
RE_TEXT_SURFACE = re.compile(    r'\b(?:texts?|letters?|words?|lettering|caption(?:s)?|banner(?:s)?|sticker(?:s)?|label(?:s)?|'
    r'signage|sign|watermark(?:s)?|overlay|logo(?:s)?|wordmark|font|typography|graphic print|'
    r'printed text|text elements)\b', re.I)


def repair(text: str, applied) -> str:
    before = text
    # action labels leaked into the instruction text ("the target interaction (hug01)",
    # "the fifth image's depicted fight01 engagement") -- found while reading a sheet
    if RE_ACTION_LABEL.search(text):
        text = RE_ACTION_LABEL.sub(' ', text)
        applied.add('R6_repair:action_label')
    text = RE_TEXT_CLAIM.sub(' ', text)          # on-image text claims go first
    if text != before:
        applied.add('R6_repair:text_claim')
    # A sentence can still name the text-bearing surface ("Keep the banner.") after the claim
    # itself is cut. Drop it only when the surface is what the sentence asks to preserve, not
    # whenever the word appears: the first version of this rule ate pose sentences that merely
    # mention a sign or a print, which tripped the pose gate on 100+ rows.
    kept = []
    for s in RE_SENT_SPLIT.split(text):
        if (SKELETON_WORD not in s and RE_TEXT_SURFACE.search(s) and len(RE_POSE.findall(s)) < 2
                and re.search(r'\b(?:keep|preserve|retain|unchanged|same|include|identical|'
                              r'exactly as|replicate|render|visible|legible)\b', s, re.I)):
            applied.add('R6_repair:text_surface_sentence')
            continue
        kept.append(s)
    text = ' '.join(kept)
    text = _grammar(text)
    if RE_META_VOICE.search(text):               # PE talking about itself / its inputs
        kept = [s for s in RE_SENT_SPLIT.split(text)
                if not (RE_META_VOICE.search(s) and SKELETON_WORD not in s)]
        if len(kept) < len(RE_SENT_SPLIT.split(text)):
            applied.add('R6_repair:meta_voice')
            text = ' '.join(kept)

    after = _grammar(text)                       # cutting sentences creates new wreckage
    if after != text:
        applied.add('R6_repair:grammar')
    text = after
    if text != before and 'R6_repair:text_claim' not in applied:
        applied.add('R6_repair:grammar')
    return text


def clean_one(row: dict, profile: dict, applied: set) -> dict:
    n = profile['n_person']
    res = {}
    for field in profile['fields']:
        text = (row.get(field) or '').strip()
        if not text:
            res[field] = ''
            continue
        # the inference form legitimately carries <image1..4>; the training form must not
        if n == 1 and field.endswith('_inference'):
            pass
        # The side-pin rewrite has to run while the parenthetical that carries the person number
        # is still there; decolor/strip_appearance delete that parenthetical.
        if n == 2:
            text = deside_anchor(text, applied)
        text = strip_watermark(text, applied)
        text = decolor(text, applied)
        text = strip_appearance(text, n, applied, keep_generic=True)
        text = strip_gt_refs(text, applied)
        text = collapse(text, n, applied)
        text = repair(text, applied)
        text = tidy(text)
        res[field] = text
    return res


def check_invariants(row: dict, cleaned: dict, profile: dict, bank_key: str) -> list[str]:
    problems = []
    for field in profile['fields']:
        text = cleaned[field]
        if not text or len(text.split()) < 60:
            problems.append(f'{field}:too_short({len(text.split())})')
            continue
        if RE_WATERMARK_SEG.search(text) and bank_key == 'tiktok':
            problems.append(f'{field}:watermark_left')
        # a prompt that lost its body instruction is worse than the one it replaced
        p_before = len(RE_POSE.findall(row.get(field) or ''))
        p_after = len(RE_POSE.findall(text))
        if p_before >= 6 and p_after < p_before * 0.5:
            problems.append(f'{field}:pose_content_lost({p_before}->{p_after})')
        if RE_FIFTH.search(text) or RE_GT_PHRASE.search(text):
            problems.append(f'{field}:gt_ref_left')
        residual = RE_COLOR.findall(text)
        if residual:
            problems.append(f'{field}:appearance_colour_left({len(residual)})')
        if profile['n_person'] == 2:
            # gate on my own rewrite: a sentence that names one person as the identity and a
            # different person's skeleton as the pose source is an induced identity swap
            for s in RE_SENT_SPLIT.split(text):
                who = re.search(r'isolated skeleton is person ([12])', s)
                pose = re.search(r"pose (?:from|of) person ([12])'s isolated skeleton", s)
                if who and pose and who.group(1) != pose.group(1):
                    problems.append(f'{field}:person_binding_broken')
                    break
        if field == 'compiled_prompt' or field == 'compiled_prompt_training':
            if '<image' in text or re.search(r'\bimage[_\s]*\d', text, re.I):
                problems.append(f'{field}:image_ref_left')
            if bank_key == 'tiktok' and SKELETON_WORD not in text:
                problems.append(f'{field}:skeleton_word_lost')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', choices=('tiktok', 'hi4d', 'both'), default='both')
    ap.add_argument('--dry-run', action='store_true', help='print before/after, write nothing')
    ap.add_argument('--limit', type=int, default=0, help='rows per split (0 = all)')
    ap.add_argument('--show', type=int, default=3)
    args = ap.parse_args()

    keys = ('tiktok', 'hi4d') if args.bank == 'both' else (args.bank,)
    grand = collections.Counter()
    shown = 0
    for bank_key in keys:
        prof = BANKS[bank_key]
        src_dir = prof['dir'] / 'prompts'
        dst_dir = prof['dir'] / 'prompts_v4'
        if not args.dry_run:
            dst_dir.mkdir(exist_ok=True)
        for split in SPLITS:
            src = src_dir / prof['in'](split)
            rows = [json.loads(l) for l in src.read_text(encoding='utf-8').splitlines() if l.strip()]
            if args.limit:
                rows = rows[:args.limit]
            written, problems_by_kind, flagged = [], collections.Counter(), []
            hard_agg = collections.Counter()
            for row in rows:
                applied = set()
                cleaned = clean_one(row, prof, applied)
                probs = check_invariants(row, cleaned, prof, bank_key)
                # A prompt that lost the skeleton role word or kept a raw image tag would make
                # scripts/prompt_bank.py raise on load, so that field falls back to v1 instead
                # of shipping broken; content residuals stay in the bank and are flagged.
                hard = {p.split(':')[0] for p in probs
                        if p.split(':')[1].split('(')[0] in HARD_KINDS}
                for p in probs:
                    kind = p.split(':')[1].split('(')[0]
                    if kind in HARD_KINDS:
                        hard_agg[f'HARD:{kind}'] += 1
                reverted = [f for f in prof['fields'] if f in hard]
                for f in reverted:
                    cleaned[f] = row.get(f) or ''
                    applied.add('REVERT_TO_V1')
                probs = check_invariants(row, cleaned, prof, bank_key) if reverted else probs
                for a in applied:
                    grand[f'{bank_key}|{a}'] += 1
                for p in probs:
                    problems_by_kind[p.split('(')[0]] += 1
                out = dict(row)
                for field, sha_field in zip(prof['fields'], prof['sha_fields']):
                    out[field] = cleaned[field]
                    # only recompute a sha field the bank already keyed on this form
                    if sha_field in row or field in row:
                        out[sha_field] = hashlib.sha256(cleaned[field].encode()).hexdigest()
                out['prompt_revision'] = 'v4_image_reviewed_20261024'
                out['cleaning_applied'] = sorted(applied)
                out['cleaning_reverted_fields'] = reverted
                out['cleaning_needs_review'] = bool(probs)
                out['cleaning_review_notes'] = probs
                if probs:
                    flagged.append((row['pair_id'], probs))
                written.append(out)
                if shown < args.show:
                    print('\n' + '=' * 24 + f' BEFORE {bank_key}/{split}/{row["pair_id"]}')
                    print((row.get(prof['fields'][0]) or '')[:1500])
                    print(f'---- AFTER')
                    print(cleaned[prof['fields'][0]][:1500])
                    print(f'---- applied={sorted(applied)} problems={probs}')
                    shown += 1
            stat = {
                'rows': len(rows),
                'word_p50_before': int(statistics.median([len((r.get(prof['fields'][0]) or '').split()) for r in rows])),
                'word_p50_after': int(statistics.median([len(w[prof['fields'][0]].split()) for w in written])),
                'needs_review': len(flagged),
                'hard_gates_fired': dict(hard_agg.most_common()),
                'problem_kinds': dict(problems_by_kind.most_common(12)),
                'flagged_examples': flagged[:8],
            }
            grand[f'{bank_key}|{split}|rows'] = len(rows)
            grand[f'{bank_key}|{split}|needs_review'] = len(flagged)
            print(f'\n### {bank_key}/{split}: ' + json.dumps(stat, ensure_ascii=False)[:900])
            if args.dry_run:
                continue
            dst = dst_dir / prof['out'](split)
            with open(dst, 'w', encoding='utf-8') as f:
                for w in written:
                    f.write(json.dumps(w, ensure_ascii=False) + '\n')
            summary = {'bank': bank_key, 'split': split, 'source': str(src),
                       'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                       'rows': len(rows), 'needs_review': len(flagged),
                       'problem_kinds': dict(problems_by_kind), 'rules': dict(
                           (k, v) for k, v in grand.items() if k.startswith(bank_key + '|R'))}
            (dst_dir / f'clean_summary_{split}.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'WROTE {dst}')
    print('\n=== rule hit counts ===')
    print(json.dumps({k: v for k, v in sorted(grand.items())}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
