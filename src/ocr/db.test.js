import { describe, expect, it } from "vitest";
import { dbPostprocess } from "./db.js";

// paint an axis-aligned block of probability p onto a [mh,mw] map
function blankMap(mh, mw) {
  return new Float32Array(mh * mw);
}
function paintRect(map, mw, x0, y0, x1, y1, p) {
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) map[y * mw + x] = p;
}

describe("dbPostprocess", () => {
  it("finds a horizontal stripe and unclips it", () => {
    const mh = 60, mw = 120;
    const map = blankMap(mh, mw);
    paintRect(map, mw, 10, 20, 99, 27, 0.95);
    const out = dbPostprocess(map, mh, mw, 1, 1, 120, 60);
    expect(out.length).toBe(1);
    const { box, score } = out[0];
    expect(score).toBeCloseTo(0.95, 6);
    // min-area rect over pixel centers is 89x7; unclip grows each side by
    // d = (89*7*1.5)/(2*(89+7)) ≈ 4.87
    const d = (89 * 7 * 1.5) / (2 * (89 + 7));
    expect(box[0]).toBeCloseTo(10 - d, 1);
    expect(box[1]).toBeCloseTo(20 - d, 1);
    expect(box[2]).toBeCloseTo(99 + d, 1);
    expect(box[3]).toBeCloseTo(27 + d, 1);
  });

  it("drops components whose mean probability is below boxThresh", () => {
    const mh = 40, mw = 80;
    const map = blankMap(mh, mw);
    paintRect(map, mw, 5, 5, 60, 12, 0.4); // binarized in (>0.3) but mean 0.4 < 0.6
    expect(dbPostprocess(map, mh, mw, 1, 1, 80, 40).length).toBe(0);
  });

  it("drops tiny components below minSize", () => {
    const mh = 40, mw = 80;
    const map = blankMap(mh, mw);
    paintRect(map, mw, 10, 10, 11, 11, 0.95); // 2x2 px
    expect(dbPostprocess(map, mh, mw, 1, 1, 80, 40).length).toBe(0);
  });

  it("separates two stripes into two boxes in any scan order", () => {
    const mh = 60, mw = 120;
    const map = blankMap(mh, mw);
    paintRect(map, mw, 10, 10, 100, 16, 0.9);
    paintRect(map, mw, 10, 35, 100, 41, 0.9);
    const out = dbPostprocess(map, mh, mw, 1, 1, 120, 60);
    expect(out.length).toBe(2);
  });

  it("recovers the angle of a rotated stripe", () => {
    const mh = 120, mw = 200;
    const map = blankMap(mh, mw);
    const deg = 15, t = Math.tan((deg * Math.PI) / 180);
    // stripe of thickness 8 along y = x*tan(15°) + 10
    for (let x = 20; x <= 150; x++) {
      const yc = Math.round(x * t) + 10;
      for (let y = yc; y < yc + 8; y++) map[y * mw + x] = 0.95;
    }
    const out = dbPostprocess(map, mh, mw, 1, 1, 200, 120);
    expect(out.length).toBe(1);
    const [p0, p1] = out[0].poly;
    const angle = (Math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180) / Math.PI;
    expect(Math.abs(angle - deg)).toBeLessThan(3);
  });

  it("scales map coordinates to original pixels and clamps", () => {
    const mh = 50, mw = 50;
    const map = blankMap(mh, mw);
    paintRect(map, mw, 0, 0, 20, 8, 0.95); // touches the border; unclip would go negative
    const out = dbPostprocess(map, mh, mw, 2, 3, 100, 150);
    expect(out.length).toBe(1);
    const { box } = out[0];
    expect(box[0]).toBe(0); // clamped
    expect(box[1]).toBe(0);
    expect(box[2]).toBeGreaterThan(20 * 2); // unclip + sx
    expect(box[3]).toBeGreaterThan(8 * 3);
  });
});
