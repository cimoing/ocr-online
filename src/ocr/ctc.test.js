import { describe, expect, it } from "vitest";
import { ctcDecode } from "./ctc.js";

const CLASSES = ["<blank>", "a", "b", " "];
const C = CLASSES.length;

// build a [T,C] prob row sequence from per-step argmax picks
function probsFor(picks) {
  const probs = new Float32Array(picks.length * C);
  picks.forEach(([cls, p], t) => {
    // spread the remainder over other classes so the pick is the argmax
    for (let c = 0; c < C; c++) probs[t * C + c] = (1 - p) / (C - 1);
    probs[t * C + cls] = p;
  });
  return probs;
}

describe("ctcDecode", () => {
  it("collapses repeats and removes blanks", () => {
    const probs = probsFor([[0, 0.9], [1, 0.9], [1, 0.8], [0, 0.9], [2, 0.8]]);
    const { text, score } = ctcDecode(probs, 0, 5, C, CLASSES);
    expect(text).toBe("ab");
    expect(score).toBeCloseTo((0.9 + 0.8) / 2, 6);
  });

  it("keeps repeated letters split by a blank", () => {
    const probs = probsFor([[1, 0.9], [0, 0.9], [1, 0.9]]);
    expect(ctcDecode(probs, 0, 3, C, CLASSES).text).toBe("aa");
  });

  it("returns empty text and zero score for all blanks", () => {
    const probs = probsFor([[0, 0.99], [0, 0.99]]);
    const { text, score } = ctcDecode(probs, 0, 2, C, CLASSES);
    expect(text).toBe("");
    expect(score).toBe(0);
  });

  it("honours the batch base offset", () => {
    const row0 = probsFor([[1, 0.9]]);
    const row1 = probsFor([[2, 0.9]]);
    const both = new Float32Array([...row0, ...row1]);
    expect(ctcDecode(both, C, 1, C, CLASSES).text).toBe("b");
  });
});
