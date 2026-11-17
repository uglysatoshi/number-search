"""Attention OCR and conservative glyph fusion; no experimental decoders."""

import torch
from torch import nn

from .constants import ALPHABET
from .geometry import tensor
from .glyphs import DIGIT_INDICES, LETTER_INDICES, segments
from .networks import AttentionEnsemble, CharacterEnsemble, decode_attention


class PlateTextReader(nn.Module):
    def __init__(self, attention_models, glyph_models, config):
        super().__init__()
        self.model = AttentionEnsemble(attention_models, list(config.temperatures))
        self.glyph_classifier = CharacterEnsemble(glyph_models)
        self.glyph_fusion_weight = config.glyph_fusion_weight
        self.recovery_log_score = config.recovery_log_score
        self.tile_recovery = config.tile_recovery

    def read_plate_crops(self, crops, batch_size=128):
        results = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                batch = crops[start : start + batch_size]
                output = self.model(torch.stack([tensor(c) for c in batch]))
                if self.glyph_fusion_weight > 0:
                    output = self.fuse_glyphs(output, batch)
                texts = decode_attention(output)
                scores = output.log_softmax(-1).max(-1).values.mean(-1).tolist()
                results.extend(
                    (text, score) if score > -0.8 else ("", -100.0)
                    for text, score in zip(texts, scores)
                )
        return results

    def fuse_glyphs(self, output, crops):
        """Conservative fusion only when a complete, well-aligned glyph segmentation exists."""
        logp = output.log_softmax(-1).clone()
        lengths = [sum((i != 0 for i in row)) for row in logp.argmax(-1).tolist()]
        candidates = []
        glyphs = []
        for crop_index, crop in enumerate(crops):
            for mode in range(4):
                boxes, items = segments(crop, mode)
                if len(items) != lengths[crop_index]:
                    continue
                if any((w / h < 0.18 or y == 0 or y + h >= 64 for x, y, w, h in boxes)):
                    continue
                bottoms = [y + h for x, y, w, h in boxes[:6]]
                if max(bottoms) - min(bottoms) > 10:
                    continue
                indices = list(range(len(glyphs), len(glyphs) + len(items)))
                glyphs.extend(items)
                candidates.append((crop_index, indices))
        if not glyphs:
            return logp
        predictions = torch.cat(
            [
                self.glyph_classifier(torch.stack(glyphs[i : i + 256])).log_softmax(-1)
                for i in range(0, len(glyphs), 256)
            ]
        ).clone()
        zero, letter_o = (ALPHABET.index("0"), ALPHABET.index("O"))
        combined = torch.logaddexp(predictions[:, zero], predictions[:, letter_o])
        predictions[:, zero] = combined
        predictions[:, letter_o] = combined
        best = {}
        for crop_index, indices in candidates:
            confidence = sum(
                (
                    float(
                        predictions[
                            index, LETTER_INDICES if j in (0, 4, 5) else DIGIT_INDICES
                        ].max()
                    )
                    for j, index in enumerate(indices)
                )
            ) / len(indices)
            if confidence < -0.1:
                continue
            if crop_index not in best or confidence > best[crop_index][0]:
                best[crop_index] = (confidence, indices)
        for crop_index, (_, indices) in best.items():
            for position, index in enumerate(indices):
                allowed = LETTER_INDICES if position in (0, 4, 5) else DIGIT_INDICES
                ids = torch.as_tensor(allowed + 1)
                scores = predictions[index, allowed]
                scores = scores - scores.logsumexp(-1)
                weight = self.glyph_fusion_weight
                logp[crop_index, position, ids] = (1 - weight) * logp[
                    crop_index, position, ids
                ] + weight * scores
        return logp
