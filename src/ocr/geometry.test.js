import { describe, expect, it } from "vitest";
import { convexHull, minAreaRect, normRect, orderCorners, rectPoly, sortReadingOrder } from "./geometry.js";

const rot = (deg) => {
  const a = (deg * Math.PI) / 180;
  return ([x, y]) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
};

// compare two point sets ignoring order/winding
function expectSamePoints(got, want, tol = 1e-6) {
  expect(got.length).toBe(want.length);
  for (const w of want) {
    const hit = got.some((g) => Math.hypot(g[0] - w[0], g[1] - w[1]) < tol);
    expect(hit, `missing point ${w}`).toBe(true);
  }
}

describe("convexHull", () => {
  it("drops interior points of a square", () => {
    const pts = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [3, 7], [9, 1]];
    expectSamePoints(convexHull(pts), [[0, 0], [10, 0], [10, 10], [0, 10]]);
  });

  it("reduces collinear points to the two endpoints", () => {
    const hull = convexHull([[0, 0], [5, 5], [10, 10], [2, 2]]);
    expectSamePoints(hull, [[0, 0], [10, 10]]);
  });
});

describe("minAreaRect", () => {
  it("recovers an axis-aligned rectangle", () => {
    const rect = minAreaRect(convexHull([[0, 0], [10, 0], [10, 4], [0, 4]]));
    expectSamePoints(rect, [[0, 0], [10, 0], [10, 4], [0, 4]], 1e-9);
  });

  it("recovers a rotated rectangle from noisy hull input", () => {
    const r = rot(30);
    const corners = [[0, 0], [20, 0], [20, 6], [0, 6]].map(r);
    const extra = [[10, 0], [20, 3], [10, 6], [0, 3]].map(r); // edge midpoints
    const rect = minAreaRect(convexHull([...corners, ...extra]));
    expectSamePoints(rect, corners, 1e-6);
  });
});

describe("orderCorners", () => {
  it("orders an axis-aligned rect as [tl,tr,br,bl]", () => {
    const out = orderCorners([[10, 4], [0, 4], [0, 0], [10, 0]]);
    expect(out).toEqual([[0, 0], [10, 0], [10, 4], [0, 4]]);
  });

  it("keeps tl->tr as the reading axis for a slightly rotated rect", () => {
    const r = rot(10);
    const corners = [[0, 0], [30, 0], [30, 8], [0, 8]].map(r);
    const [tl, tr, , bl] = orderCorners([corners[2], corners[0], corners[3], corners[1]]);
    expect(tr[0]).toBeGreaterThan(tl[0]); // reading direction goes right
    expect(bl[1]).toBeGreaterThan(tl[1]); // and down the page
  });
});

describe("normRect", () => {
  it("normalizes swapped corners", () => {
    expect(normRect(50, 40, 10, 8, 100, 100, 4)).toEqual([10, 8, 50, 40]);
  });

  it("clamps to the image and enforces a minimum size", () => {
    expect(normRect(-5, -5, 1, 1, 100, 100, 4)).toEqual([0, 0, 4, 4]);
    expect(normRect(98, 98, 99, 99, 100, 100, 4)).toEqual([96, 96, 100, 100]);
  });
});

describe("sortReadingOrder", () => {
  it("sorts by 10px row bands then x", () => {
    const entries = [
      { box: [50, 12, 60, 20], id: "b" },
      { box: [5, 14, 15, 22], id: "a" },
      { box: [0, 40, 10, 48], id: "c" },
    ];
    sortReadingOrder(entries);
    expect(entries.map((e) => e.id)).toEqual(["a", "b", "c"]);
  });
});

describe("rectPoly", () => {
  it("expands a box to [tl,tr,br,bl]", () => {
    expect(rectPoly([1, 2, 3, 4])).toEqual([[1, 2], [3, 2], [3, 4], [1, 4]]);
  });
});
