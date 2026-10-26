// Research presentation: compact latent visual policies + VLM programmer.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "paper", "figures", f);
const read = (p, d) => { try { return JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8")); } catch { return d; } };

// ---------- data from result files ----------
const abl = {};
for (const v of ["both16", "head16", "wrist16", "both4", "both64"]) {
  const runs = [read(`results/ablations/${v}.json`), read(`results/ablations/${v}_s8.json`), read(`results/ablations/${v}_s9.json`)].filter(Boolean);
  const mean = (k) => runs.reduce((a, r) => a + r.val[k], 0) / runs.length;
  abl[v] = { phase: mean("phase_acc"), act: mean("act_l1_scaled"), n: runs.length };
}
const vlm = read("results/vlm_verify.json", null);
const h2h = { direct: read("results/h2h_direct.json", []), skill: read("results/h2h_skill.json", []) };
const h2hSum = (rs) => {
  const ok = rs.filter((r) => r.success);
  return { ok: ok.length, n: rs.length, t: ok.length ? ok.reduce((a, r) => a + r.seconds, 0) / ok.length : null,
           calls: rs.length ? rs.reduce((a, r) => a + r.opus_calls, 0) / rs.length : 0 };
};

const THEME = {
  name: "Latent Vision",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F24", lt1: "FFFFFF", dk2: "2B3138", lt2: "EEF1F4",
    accent1: "E07B28", accent2: "2F6DB5", accent3: "3A9A5B", accent4: "7A4FB5", accent5: "8A8F98", accent6: "C4483E",
    hlink: "2F6DB5", folHlink: "7A4FB5",
  },
};

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "How Small Can a Robot's Eyes Be?";
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "Dark",
    background: { color: THEME.colors.dk1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.3, w: 7.4, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.35, w: 7.2, h: 1.6, fontSize: 18, color: "C9CFD6", valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "Content",
    background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left" }, text: "" } },
    ],
  });

  const T = (slide, text, opts) => slide.addText(text, { isTextBox: true, fontSize: 16, color: C.text2, margin: 0, valign: "top", ...opts });
  const card = (slide, x, y, w, h, name) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: "none" }, objectName: name });
  const stat = (slide, x, y, big, small, color) => {
    T(slide, big, { x, y, w: 2.9, h: 0.9, fontSize: 48, bold: true, color, fontFace: THEME.headFontFace });
    T(slide, small, { x, y: y + 0.95, w: 2.9, h: 0.8, fontSize: 14, color: C.text2 });
  };

  // 1 ---------- title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "Dark", sectionTitle: "Introduction" });
  s.addImage({ path: path.join(ROOT, "results", "vlm_states", "state03_overview.png"), x: 7.9, y: 0, w: 5.43, h: 7.5, sizing: { type: "cover", w: 5.43, h: 7.5 }, objectName: "hero" });
  s.addText("How Small Can a Robot's Eyes Be?", { placeholder: "title" });
  s.addText("Compact latent visual policies and a vision-language programmer for camera-only mobile manipulation", { placeholder: "body" });
  T(s, "Author Name  ·  Machine Vision", { x: 0.7, y: 6.55, w: 7, h: 0.4, fontSize: 14, color: "8A9099" });
  s.addNotes("Question of the talk: how much visual information does a robot policy need if everything it sees is squeezed into 16 numbers?");

  // 2 ---------- the question / numbers
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Introduction" });
  s.addText("Everything the policy sees fits in 16 numbers", { placeholder: "title" });
  stat(s, 0.6, 1.7, "16", "dimensional latent vector: the policy's entire view of the world", C.accent1);
  stat(s, 3.75, 1.7, "255k", "parameters in the whole network", C.accent2);
  stat(s, 6.9, 1.7, "1.5 ms", "per decision for six arms (CUDA graph, idle GPU)", C.accent3);
  stat(s, 10.05, 1.7, "2", "low-resolution RGB cameras: head 96×96, wrist 64×64", C.accent4);
  card(s, 0.6, 4.3, 12.1, 2.05, "question-card");
  T(s, [
    { text: "Research questions", options: { bold: true, color: C.text1, fontSize: 18, breakLine: true } },
    { text: "What task structure survives a 16-D visual bottleneck?", options: { bullet: true, breakLine: true } },
    { text: "Which skills can such a policy perform on its own, and which need a slower, smarter layer above it?", options: { bullet: true, breakLine: true } },
    { text: "Can a vision-language model program and verify those skills from camera images?", options: { bullet: true } },
  ], { x: 0.9, y: 4.5, w: 11.5, h: 1.75, paraSpaceAfter: 6 });

  // 3 ---------- scene
  pres.addSection({ title: "System" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "System" });
  s.addText("Simulated cell: pick, drive, drop", { placeholder: "title" });
  s.addImage({ path: FIG("scene.png"), x: 0.6, y: 1.45, w: 8.4, h: 2.82, objectName: "scene" });
  T(s, [
    { text: "Isaac Sim 5.1, PhysX 120 Hz, control 15 Hz", options: { bullet: true, breakLine: true } },
    { text: "Ridgeback + Franka Panda; holonomic base model", options: { bullet: true, breakLine: true } },
    { text: "Red cube, blue ball, green puck; three bins", options: { bullet: true, breakLine: true } },
    { text: "Friction grasps only: no welds, no teleporting", options: { bullet: true } },
  ], { x: 9.3, y: 1.55, w: 3.6, h: 2.8, fontSize: 15, paraSpaceAfter: 8 });
  card(s, 0.6, 4.65, 12.1, 1.55, "teacher-card");
  T(s, [
    { text: "Privileged teacher (training data only)", options: { bold: true, color: C.text1, fontSize: 17, breakLine: true } },
    { text: "Reads true poses from the simulator: grid path planning for driving, a descent funnel for grasping. Its decisions depend only on the current state, so it can relabel any state the student reaches. It never drives a reported student run.", options: {} },
  ], { x: 0.9, y: 4.85, w: 11.5, h: 1.2, fontSize: 15 });

  // 4 ---------- hierarchy
  s = pres.addSlide({ masterName: "Content", sectionTitle: "System" });
  s.addText("Two speeds: a VLM writes programs, a tiny policy moves", { placeholder: "title" });
  const lvl = [
    ["Claude Opus 5.5 programmer", "reads 4 camera views + execution log, writes face / goto / pick / drop programs, verifies outcomes", "~10 s per call", C.accent4],
    ["Learned latent skills", "pick(color), drop(): gripper x/y/z velocity + grip from two RGB cameras", "15 Hz, 1.5–15 ms", C.accent1],
    ["Fixed controllers", "face(): heading control · goto(): grid planner · wrist angle hold · inverse kinematics", "15 Hz", C.accent5],
  ];
  lvl.forEach(([name, desc, rate, col], i) => {
    const y = 1.6 + i * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 1.45, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: col, width: 1.5 }, objectName: `level-${i}` });
    T(s, name, { x: 0.9, y: y + 0.18, w: 4.0, h: 0.5, fontSize: 19, bold: true, color: col });
    T(s, desc, { x: 0.9, y: y + 0.68, w: 8.6, h: 0.65, fontSize: 15 });
    T(s, rate, { x: 9.8, y: y + 0.45, w: 2.7, h: 0.5, fontSize: 18, bold: true, color: C.text1, align: "right" });
  });

  // 4b ---------- roles
  s = pres.addSlide({ masterName: "Content", sectionTitle: "System" });
  s.addText("Three roles: teacher, student, operator", { placeholder: "title" });
  const roles = [
    ["Teacher", "scripted controller", "Reads exact object poses from the simulator. Demonstrates the task and labels the student's mistakes. Used only to make training data.", "training only", C.accent5],
    ["Student", "16-D latent policy", "Learns to copy the teacher from two camera images alone, without the poses. Becomes the pick and drop skills.", "learned", C.accent1],
    ["Operator", "Claude Opus 5.5", "Calls the trained skills, looks at the cameras, checks the outcome and retries. Never sees the teacher or the training data.", "run time", C.accent4],
  ];
  roles.forEach(([name, who, what, when, col], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 3.85, h: 3.6, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: col, width: 1.5 }, objectName: `role-${name}` });
    T(s, name, { x: x + 0.3, y: 1.8, w: 3.3, h: 0.55, fontSize: 24, bold: true, color: col, fontFace: THEME.headFontFace });
    T(s, who, { x: x + 0.3, y: 2.38, w: 3.3, h: 0.4, fontSize: 15, bold: true, color: C.text1 });
    T(s, what, { x: x + 0.3, y: 2.9, w: 3.3, h: 1.8, fontSize: 14 });
    T(s, when, { x: x + 0.3, y: 4.7, w: 3.3, h: 0.35, fontSize: 12, bold: true, color: "6B7280" });
    if (i < 2) T(s, "→", { x: x + 3.85, y: 3.1, w: 0.25, h: 0.5, fontSize: 20, bold: true, color: "8A8F98", align: "center" });
  });
  card(s, 0.6, 5.5, 12.1, 1.05, "roles-note");
  T(s, [
    { text: "Teacher → student is imitation learning; student → operator is a function call. ", options: { bold: true, color: C.text1 } },
    { text: "The teacher can cheat because the simulator knows where everything is. The student has to learn the same behaviour from pixels, which is what a real robot would have." },
  ], { x: 0.9, y: 5.68, w: 11.5, h: 0.75, fontSize: 15 });

  // 5 ---------- architecture
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Method" });
  s.addText("Latent mixture-of-experts policy", { placeholder: "title" });
  const box = (x, y, w, h, text, fill, line, name, bold) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: line, width: 1.25 }, objectName: name });
    T(s, text, { x, y, w, h, align: "center", valign: "middle", fontSize: 14, color: C.text1, bold: !!bold });
  };
  const arrow = (x1, y1, x2, y2, name) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, flipV: y2 < y1, line: { color: "6B7280", width: 1.25, endArrowType: "triangle" }, objectName: name });
  box(0.6, 1.7, 1.9, 0.9, "head RGB\n96×96 ×2 frames", "E8F0FB", "2F6DB5", "in-head");
  box(0.6, 3.1, 1.9, 0.9, "wrist RGB\n64×64 ×2 frames", "E8F0FB", "2F6DB5", "in-wrist");
  box(3.0, 1.7, 2.3, 0.9, "depthwise CNN +\nspatial softmax", "F4F5F7", "8A8F98", "enc-head");
  box(3.0, 3.1, 2.3, 0.9, "depthwise CNN +\nspatial softmax", "F4F5F7", "8A8F98", "enc-wrist");
  box(5.8, 2.2, 1.6, 1.3, "z ∈ ℝ¹⁶\nlatent", "FDF0E4", "E07B28", "latent", true);
  box(0.6, 4.6, 4.7, 0.8, "joints · gripper width · base velocity · skill command", "F4F5F7", "8A8F98", "proprio");
  box(7.9, 1.55, 1.6, 0.7, "gate", "F4F5F7", "8A8F98", "gate");
  box(7.9, 2.55, 1.6, 1.3, "4 experts", "EAF6EE", "3A9A5B", "experts");
  box(10.0, 2.45, 2.7, 1.5, "action chunk K=4\nbase v · gripper v · grip", "F4F5F7", "8A8F98", "out");
  box(5.8, 4.6, 2.4, 0.8, "phase head (from z)", "F4F5F7", "8A8F98", "phase");
  arrow(2.5, 2.15, 3.0, 2.15, "a1"); arrow(2.5, 3.55, 3.0, 3.55, "a2"); arrow(5.3, 2.15, 5.8, 2.6, "a3"); arrow(5.3, 3.55, 5.8, 3.1, "a4");
  arrow(7.4, 2.6, 7.9, 1.9, "a5"); arrow(7.4, 3.0, 7.9, 3.2, "a6"); arrow(9.5, 3.2, 10.0, 3.2, "a7"); arrow(6.6, 3.5, 6.9, 4.6, "a8"); arrow(5.3, 5.0, 7.9, 3.85, "a9");
  T(s, [
    { text: "255,249 parameters · two-thirds in the experts", options: { bullet: true, breakLine: true } },
    { text: "Rotation is never an output: heading and wrist angle are held by fixed controllers", options: { bullet: true, breakLine: true } },
    { text: "Whole forward pass captured as one CUDA graph: 7.2 ms → 1.5 ms per decision", options: { bullet: true, breakLine: true } },
    { text: "Not PCA: z is trained with the action, so it keeps what the robot needs to act, not what varies most in the pixels", options: { bullet: true } },
  ], { x: 0.6, y: 5.6, w: 12.1, h: 1.7, fontSize: 15, paraSpaceAfter: 3 });

  // 6 ---------- training recipe
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Method" });
  s.addText("Training: imitation, then let the student make mistakes", { placeholder: "title" });
  const steps = [["1", "Teacher episodes", "scripted controller with true poses; Gaussian action noise (DART) so recoveries appear"],
                 ["2", "Behaviour cloning", "smooth-L1 on action chunks, BCE on grip, phase CE, gate balance; ±4 px shift + colour jitter"],
                 ["3", "DAgger rounds", "student executes each step with probability 1−β; teacher labels every visited state"],
                 ["4", "Specialists", "pick-focused and drive-focused rounds; the executor routes each skill to its checkpoint"]];
  steps.forEach(([n, head, body], i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.OVAL, { x, y: 1.75, w: 0.75, h: 0.75, fill: { color: C.accent1 }, line: { type: "none" }, objectName: `step-${n}` });
    T(s, n, { x, y: 1.75, w: 0.75, h: 0.75, align: "center", valign: "middle", fontSize: 22, bold: true, color: C.background1 });
    T(s, head, { x, y: 2.75, w: 2.8, h: 0.5, fontSize: 18, bold: true, color: C.text1 });
    T(s, body, { x, y: 3.3, w: 2.8, h: 1.6, fontSize: 14 });
  });
  card(s, 0.6, 5.15, 12.1, 1.4, "action-card");
  T(s, [
    { text: "Action space mattered more than architecture", options: { bold: true, color: C.text1, fontSize: 17, breakLine: true } },
    { text: "With joint-angle deltas, a 5 mm grasp correction is ~0.005 rad, the size of the network's own error, so the student never closed on objects. Gripper velocity in the arm frame turns the same correction into a learnable signal.", options: {} },
  ], { x: 0.9, y: 5.32, w: 11.5, h: 1.1, fontSize: 15 });

  // 7 ---------- data pipeline failures
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("The cameras lied before the model did", { placeholder: "title" });
  s.addImage({ path: FIG("frame_integrity.png"), x: 0.6, y: 1.5, w: 6.4, h: 3.77, objectName: "integrity" });
  s.addImage({ path: FIG("flat_example.png"), x: 1.6, y: 5.35, w: 3.6, h: 1.94, objectName: "flat" });
  const fails = [["~1 m near clip", "default camera clipping hid the table, floor and arm right in front of the robot"],
                 ["Unfinished frames", "two renderers on one GPU: 100% / 54% sky-only frames; 47% in one DAgger round"],
                 ["Black cameras", "120 pick episodes with a black wrist camera, silently dropped by the training filter"]];
  fails.forEach(([h, b], i) => {
    const y = 1.55 + i * 1.85;
    card(s, 7.4, y, 5.3, 1.6, `fail-${i}`);
    T(s, h, { x: 7.65, y: y + 0.15, w: 4.9, h: 0.45, fontSize: 18, bold: true, color: C.accent6 });
    T(s, b, { x: 7.65, y: y + 0.65, w: 4.9, h: 0.85, fontSize: 14 });
  });

  // 8 ---------- latent
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("The 16-D latent organises the task", { placeholder: "title" });
  s.addImage({ path: FIG("latent_tsne.png"), x: 0.6, y: 1.4, w: 6.3, h: 5.2, sizing: { type: "contain", w: 6.3, h: 5.2 }, objectName: "tsne" });
  s.addImage({ path: FIG("phase_confusion.png"), x: 7.1, y: 1.4, w: 5.6, h: 4.65, sizing: { type: "contain", w: 5.6, h: 4.65 }, objectName: "confusion" });
  T(s, "Driving and manipulation separate; pick phases form an ordered curve. A linear head on z recovers the teacher's phase on 95% of frames, confusing only neighbouring phases.", { x: 7.1, y: 6.15, w: 5.6, h: 1.0, fontSize: 14 });

  // 9 ---------- keypoints
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Where the policy looks", { placeholder: "title" });
  s.addImage({ path: FIG("keypoints.png"), x: 1.4, y: 1.35, w: 7.8, h: 5.4, sizing: { type: "contain", w: 7.8, h: 5.4 }, objectName: "keypoints" });
  T(s, [
    { text: "Spatial-softmax keypoints", options: { bold: true, color: C.text1, fontSize: 18, breakLine: true } },
    { text: "Wrist view: object, fingertips, table edge", options: { bullet: true, breakLine: true } },
    { text: "Head view: table edge and objects on it", options: { bullet: true, breakLine: true } },
    { text: "Dot size = how peaked the channel is", options: { bullet: true } },
  ], { x: 9.6, y: 2.2, w: 3.3, h: 3.0, fontSize: 15, paraSpaceAfter: 8 });

  // 10 ---------- experts (native chart)
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Experts specialise by skill without being told", { placeholder: "title" });
  const gate = read("paper/vision_stats.json", { gate_usage: { goto: [0, 0, 0, 0], pick: [0, 0, 0, 0], drop: [0, 0, 0, 0] } }).gate_usage;
  const skills = ["goto", "pick", "drop"];
  s.addChart(pres.charts.BAR, [0, 1, 2, 3].map((e) => ({ name: `expert ${e + 1}`, labels: skills, values: skills.map((k) => +(gate[k][e] * 100).toFixed(1)) })), {
    x: 0.6, y: 1.45, w: 7.4, h: 5.4, barDir: "col", barGrouping: "stacked", chartColors: ["8A8F98", "2F6DB5", "E07B28", "3A9A5B"],
    showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: "2B3138",
    showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0", dataLabelColor: "FFFFFF", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: "2B3138", catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 13, valAxisLabelColor: "6B7280", valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 11,
    valAxisMaxVal: 100, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
    showTitle: true, title: "mean gate weight per skill (%)", titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: "2B3138",
  });
  stat(s, 8.6, 1.7, `${Math.round(gate.pick[0] * 100)}%`, "of pick routed to expert 1", C.accent3);
  stat(s, 8.6, 3.5, `${Math.round(gate.drop[2] * 100)}%`, "of drop routed to expert 3", C.accent1);
  T(s, "The skill command is an input, so the gate could route on it; it does, while the latent stays shared across skills.", { x: 8.6, y: 5.35, w: 4.1, h: 1.3, fontSize: 14 });

  // 11 ---------- ablations (native chart)
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Ablations: wrist camera and latent size", { placeholder: "title" });
  const order = [["both16", "both cameras, 16-D"], ["head16", "head only, 16-D"], ["wrist16", "wrist only, 16-D"], ["both4", "both, 4-D"], ["both64", "both, 64-D"]];
  s.addChart(pres.charts.BAR, [{ name: "phase accuracy", labels: order.map((o) => o[1]), values: order.map((o) => +(abl[o[0]].phase * 100).toFixed(1)) }], {
    x: 0.6, y: 1.45, w: 7.6, h: 5.4, barDir: "bar", chartColors: ["2F6DB5"], showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelColor: "2B3138", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: "2B3138", catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 14, valAxisLabelColor: "6B7280", valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 12,
    valAxisMinVal: 70, valAxisMaxVal: 100, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, catAxisOrientation: "maxMin",
    showTitle: true, title: `held-out phase accuracy (%), mean of ${abl.both16.n} seeds`, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: "2B3138",
  });
  T(s, [
    { text: "What it says", options: { bold: true, color: C.text1, fontSize: 18, breakLine: true } },
    { text: "The wrist camera carries most of the task information; head-only loses the most", options: { bullet: true, breakLine: true } },
    { text: "4-D is too small; 64-D helps a little offline", options: { bullet: true, breakLine: true } },
    { text: "Offline numbers only: they do not predict closed-loop success", options: { bullet: true } },
  ], { x: 8.7, y: 1.8, w: 4.1, h: 4.6, fontSize: 15, paraSpaceAfter: 10 });

  // 12 ---------- closed loop
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Closed loop: skills work alone, the monolith never did", { placeholder: "title" });
  s.addImage({ path: FIG("dagger_curves.png"), x: 0.6, y: 1.4, w: 7.0, h: 4.73, objectName: "dagger" });
  T(s, "Success vs share of steps executed by the student (95% Wilson intervals). Points at 1.0 are pure student control.", { x: 0.6, y: 6.2, w: 7.0, h: 0.8, fontSize: 12, color: "6B7280" });
  stat(s, 8.2, 1.55, "14/20", "camera-only pick, no teacher at all", C.accent2);
  stat(s, 8.2, 3.25, "4/8", "full deliveries: learned pick + drop, planner goto", C.accent3);
  stat(s, 8.2, 4.95, "0", "solo successes of the end-to-end delivery policy", C.accent6);

  // 13 ---------- VLM programmer + verification
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Opus as programmer and verifier", { placeholder: "title" });
  card(s, 0.6, 1.5, 6.0, 3.6, "program-card");
  T(s, [
    { text: "Recorded run", options: { bold: true, color: C.text1, fontSize: 18, breakLine: true } },
    { text: "pick(blue)  face(east)  goto(teal)  drop()", options: { fontFace: "Courier New", fontSize: 14, color: C.accent4, breakLine: true } },
    { text: "Verified from images that the ball left the table", options: { bullet: true, breakLine: true } },
    { text: "face(west)  goto(table)  face(north)  pick(red)", options: { fontFace: "Courier New", fontSize: 14, color: C.accent4, breakLine: true } },
    { text: "Retried pick(red) after a timeout", options: { bullet: true, breakLine: true } },
    { text: "Then stopped, wrongly concluding the cube had left the table: its table and bin cameras had rendered black", options: { bullet: true } },
  ], { x: 0.9, y: 1.7, w: 5.5, h: 5.0, fontSize: 15, paraSpaceAfter: 7 });
  if (vlm && vlm.summary) {
    const sm = vlm.summary;
    const rows = [["Views given", "Accuracy", "Table", "In bin", "Floor"]];
    for (const [k, label] of [["table+bins+overview", "table + bins + overview"], ["overview", "overview only"]]) {
      const v = sm[k]; const p = v.per_true_location || {};
      const bins = ["teal", "magenta", "gold"].map((b) => p[b] || "0/0").reduce((a, x) => { const [n, d] = x.split("/").map(Number); return [a[0] + n, a[1] + d]; }, [0, 0]);
      rows.push([label, `${(v.accuracy * 100).toFixed(0)}%`, p.table || "–", `${bins[0]}/${bins[1]}`, p.floor || "–"]);
    }
    s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: "1B1F24", fontSize: 14, fill: { color: i === 0 ? "EEF1F4" : "FFFFFF" } } }))), {
      x: 6.9, y: 1.6, w: 5.8, colW: [2.2, 1.0, 0.85, 0.9, 0.85], border: { type: "solid", pt: 0.5, color: "D1D5DB" }, rowH: 0.45, objectName: "vlm-table",
    });
    T(s, `Object locations from rendered images vs PhysX ground truth: 24 scenes × 3 objects. Errors are almost all "not visible" abstentions for objects inside bins; only one answer named a wrong location.`, { x: 6.9, y: 3.25, w: 5.8, h: 0.9, fontSize: 13, color: "6B7280" });
  }
  s.addMedia({ type: "video", path: path.join(ROOT, "artifacts", "demo3_opus_programs_robot_small.mp4"), cover: "data:image/png;base64," + fs.readFileSync(path.join(ROOT, "paper", "figures", "video_cover.png")).toString("base64"), x: 6.9, y: 4.3, w: 4.3, h: 2.42, objectName: "demo-video" });

  T(s, "Disclosure: the table/bin layout and the pick → face → goto → drop recipe are written in Opus's prompt; it chooses objects, order, checks and retries.", { x: 0.6, y: 6.85, w: 6.2, h: 0.5, fontSize: 11, color: "6B7280" });

  // 13b ---------- deployment head-to-head
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Deployment test: Opus alone vs Opus + 16-D skill", { placeholder: "title" });
  T(s, "Same robot, same two cameras, same instruction (\"pick up the <colour> object\"), same Opus model and prompt budget. Success is read from PhysX: the commanded object held 5 cm above the table. Clocks are wall time, including Opus thinking.", { x: 0.6, y: 1.35, w: 12.1, h: 0.75, fontSize: 15 });
  const hd = h2hSum(h2h.direct), hs = h2hSum(h2h.skill);
  const col = (x, title, color, r, how) => {
    card(s, x, 2.3, 3.9, 4.55, `h2h-${title}`);
    T(s, title, { x: x + 0.3, y: 2.5, w: 3.4, h: 0.45, fontSize: 18, bold: true, color });
    T(s, how, { x: x + 0.3, y: 2.95, w: 3.4, h: 0.7, fontSize: 12, color: "6B7280" });
    T(s, `${r.ok}/${r.n}`, { x: x + 0.3, y: 3.65, w: 3.4, h: 0.85, fontSize: 44, bold: true, color: C.text1, fontFace: THEME.headFontFace });
    T(s, "successful picks", { x: x + 0.3, y: 4.45, w: 3.4, h: 0.35, fontSize: 13 });
    T(s, r.t === null ? "never" : `${r.t.toFixed(0)} s`, { x: x + 0.3, y: 4.9, w: 3.4, h: 0.7, fontSize: 32, bold: true, color: C.text1, fontFace: THEME.headFontFace });
    T(s, "mean time when successful", { x: x + 0.3, y: 5.55, w: 3.4, h: 0.35, fontSize: 13 });
    T(s, `${r.calls.toFixed(1)} Opus calls per attempt`, { x: x + 0.3, y: 6.05, w: 3.4, h: 0.5, fontSize: 15, bold: true });
  };
  col(0.6, "Opus alone", C.accent1, hd, "each call moves the gripper ≤ 8 cm and sets open/close");
  col(4.7, "Opus + 16-D skill", C.accent3, hs, "Opus calls pick(colour); the latent policy closes the loop at 15 Hz");
  s.addMedia({ type: "video", path: path.join(ROOT, "deliverables", "claude_vs_claude_plus_16d.mp4"), cover: "data:image/png;base64," + fs.readFileSync(path.join(ROOT, "paper", "figures", "h2h_cover.png")).toString("base64"), x: 8.85, y: 2.3, w: 3.9, h: 2.19, objectName: "h2h-video" });
  T(s, "Comparison video (under 60 s), real time then labelled speed-up", { x: 8.85, y: 4.6, w: 3.9, h: 0.6, fontSize: 12, color: "6B7280" });
  T(s, "A language model reasons well but sees and acts once every ~10 s; contact-rich grasping needs feedback every ~70 ms.", { x: 8.85, y: 5.3, w: 3.9, h: 1.5, fontSize: 15, bold: true, color: C.text1 });

  // 14 ---------- latency
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Results" });
  s.addText("Four orders of magnitude between the two layers", { placeholder: "title" });
  s.addImage({ path: FIG("latency.png"), x: 0.6, y: 1.5, w: 7.6, h: 4.26, objectName: "latency" });
  T(s, [
    { text: "Inside the running simulator a decision takes 6–15 ms, well under the 67 ms control period", options: { bullet: true, breakLine: true } },
    { text: "A VLM call takes 6–13 s, so it can only act on programs, never on motor commands", options: { bullet: true, breakLine: true } },
    { text: "The robot holds still while Opus thinks", options: { bullet: true } },
  ], { x: 8.7, y: 1.8, w: 4.1, h: 4.4, fontSize: 15, paraSpaceAfter: 10 });

  // 15 ---------- limitations
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "Content", sectionTitle: "Conclusion" });
  s.addText("Where it fits, and the next step", { placeholder: "title" });
  card(s, 0.6, 1.5, 5.95, 2.45, "fits-card");
  T(s, [
    { text: "Fits now: fixed, structured cells", options: { bold: true, color: C.accent3, fontSize: 18, breakLine: true } },
    { text: "Factory stations, warehouses, lab automation: one setting, known parts", options: { bullet: true, breakLine: true } },
    { text: "A skill takes a few hours, mostly simulator time; training itself is minutes", options: { bullet: true, breakLine: true } },
    { text: "1.5 ms decisions; 7.5 ms even on a CPU", options: { bullet: true } },
  ], { x: 0.9, y: 1.7, w: 5.4, h: 2.15, fontSize: 15, paraSpaceAfter: 5 });
  card(s, 6.75, 1.5, 5.95, 2.45, "needs-card");
  T(s, [
    { text: "Needs work: the open world", options: { bold: true, color: C.accent6, fontSize: 18, breakLine: true } },
    { text: "Varied scenes and objects; learned navigation (1/10 alone here)", options: { bullet: true, breakLine: true } },
    { text: "Memory and world models: the policy sees only the current frame", options: { bullet: true, breakLine: true } },
    { text: "Sim-to-real transfer; larger latents (64-D already beat 16-D)", options: { bullet: true } },
  ], { x: 7.05, y: 1.7, w: 5.4, h: 2.15, fontSize: 15, paraSpaceAfter: 5 });
  T(s, "Next: one Opus that builds its own skills, then uses them", { x: 0.6, y: 4.2, w: 12.1, h: 0.5, fontSize: 19, bold: true, color: C.text1 });
  const flow = ["describe task", "Opus writes the teacher", "collect demos", "train 16-D student", "verify in physics", "add to toolbox, operate"];
  flow.forEach((t, i) => {
    const x = 0.6 + i * 2.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.85, w: 1.8, h: 0.85, rectRadius: 0.08, fill: { color: i === 1 ? "F3EEFB" : C.background2 }, line: { color: i === 1 ? "7A4FB5" : "C9CFD6", width: 1 }, objectName: `flow-${i}` });
    T(s, t, { x: x + 0.12, y: 4.85, w: 1.56, h: 0.85, fontSize: 13, align: "center", valign: "middle", color: C.text1 });
    if (i < flow.length - 1) T(s, "→", { x: x + 1.8, y: 5.05, w: 0.25, h: 0.45, fontSize: 16, color: "8A8F98", align: "center" });
  });
  T(s, "Today these are two separate Opus roles: an engineer (a coding assistant that wrote the teacher, simulator and training code with the author) and an operator (calls skills at run time). The deployed system does not write its own teachers yet; merging the two roles is the next step, with a human still checking each new skill.", { x: 0.6, y: 5.95, w: 12.1, h: 1.0, fontSize: 13, color: "6B7280" });


  s = pres.addSlide({ masterName: "Content", sectionTitle: "Conclusion" });
  s.addText("Limitations", { placeholder: "title" });
  const lims = [["Simulation only", "holonomic base model, one fixed arena, nearly fixed object positions"],
                ["Small evaluations", "8–20 closed-loop trials per condition; wide confidence intervals"],
                ["Navigation not learned", "learned goto: 34/36 with help, 1/10 alone; replaced by a planner"],
                ["Offline ablations", "single training budget, three seeds; no closed-loop ablation"]];
  lims.forEach(([h, b], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.7 + Math.floor(i / 2) * 2.3;
    card(s, x, y, 5.95, 1.9, `lim-${i}`);
    T(s, h, { x: x + 0.3, y: y + 0.3, w: 5.4, h: 0.5, fontSize: 19, bold: true, color: C.text1 });
    T(s, b, { x: x + 0.3, y: y + 0.9, w: 5.4, h: 0.8, fontSize: 16 });
  });

  // 16 ---------- conclusion
  s = pres.addSlide({ masterName: "Dark", sectionTitle: "Conclusion" });
  s.addText("Small eyes, big planner", { placeholder: "title" });
  s.addText([
    { text: "16 numbers capture the task's structure and drive a camera-only pick 14/20 times", options: { bullet: true, breakLine: true } },
    { text: "Experts specialise by skill on their own", options: { bullet: true, breakLine: true } },
    { text: "Complete tasks need a hierarchy at this size: a VLM that programs, verifies and retries", options: { bullet: true, breakLine: true } },
    { text: `Deployment test: Opus alone picks ${h2hSum(h2h.direct).ok}/${h2hSum(h2h.direct).n}, with the 16-D skill ${h2hSum(h2h.skill).ok}/${h2hSum(h2h.skill).n}`, options: { bullet: true, breakLine: true } },
    { text: "Check the cameras before blaming the model", options: { bullet: true } },
  ], { placeholder: "body", paraSpaceAfter: 6 });
  s.addMedia({ type: "video", path: path.join(ROOT, "artifacts", "demo2_skills_best_run_small.mp4"), cover: "data:image/png;base64," + fs.readFileSync(path.join(ROOT, "paper", "figures", "video_cover2.png")).toString("base64"), x: 8.3, y: 2.2, w: 4.5, h: 2.53, objectName: "best-run-video" });
  T(s, "Best recorded run: two deliveries by the learned pick and drop skills (real time)", { x: 8.3, y: 4.85, w: 4.5, h: 0.7, fontSize: 13, color: "8A9099" });

  const out = path.join(__dirname, "latent_vision_presentation.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
