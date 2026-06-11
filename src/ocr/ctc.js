// CTC greedy decode over a softmax probability sequence.
// `probs` is a flat [.., T, C] buffer; `base` selects the batch row.
// Class 0 is the CTC blank; repeated argmaxes collapse.
export function ctcDecode(probs, base, T, C, classes) {
  let prev = -1;
  let text = "";
  let scoreSum = 0, scoreN = 0;
  for (let t = 0; t < T; t++) {
    const off = base + t * C;
    let best = 0, bestVal = probs[off];
    for (let c = 1; c < C; c++) {
      const v = probs[off + c];
      if (v > bestVal) { bestVal = v; best = c; }
    }
    if (best !== prev && best !== 0) {
      text += classes[best];
      scoreSum += bestVal; // already softmax probabilities -> use directly
      scoreN++;
    }
    prev = best;
  }
  return { text, score: scoreN ? scoreSum / scoreN : 0 };
}
