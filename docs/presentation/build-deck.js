const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");
// The pptx skill's apply_theme.js writes the theme colours; without it the deck keeps Office's defaults.
let applyTheme = async () => {};
try { ({ applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js")); } catch { console.warn("apply_theme.js not found: default theme colours"); }

const SHOTS = path.join(__dirname, "screenshots");
const OUT = path.join(__dirname, "TestPoint-management-2026-11-02.pptx");

const THEME = {
  name: "TestPoint",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "16232B", lt1: "FFFFFF", dk2: "3C4B55", lt2: "EEF2F1",
    accent1: "1F7A5A", accent2: "E8A33D", accent3: "2E86C1", accent4: "C0392B",
    accent5: "7F8C8D", accent6: "0E3B2E", hlink: "1F7A5A", folHlink: "0E3B2E",
  },
};

async function img(file, crop) {
  let s = sharp(file);
  if (crop) {
    const m = await sharp(file).metadata();
    s = s.extract({ left: Math.round(crop[0] * m.width), top: Math.round(crop[1] * m.height),
                    width: Math.round(crop[2] * m.width), height: Math.round(crop[3] * m.height) });
  }
  const buf = await s.resize({ width: 1800, withoutEnlargement: true }).png().toBuffer();
  const m = await sharp(buf).metadata();
  return { data: "image/png;base64," + buf.toString("base64"), r: m.width / m.height };
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.rtlMode = true;
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "TestPoint - عرض الإدارة";
  const C = pres.SchemeColor;
  const AR = { rtlMode: true, lang: "ar-SA" };

  pres.defineSlideMaster({
    title: "DARK", background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.4, fontSize: 46, bold: true, color: C.background1, align: "right", valign: "bottom", ...AR }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 2.9, fontSize: 22, color: C.accent2, align: "right", valign: "top", ...AR }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 }, margin: [0.5, 0.6, 0.5, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 11.0, h: 0.9, fontSize: 32, bold: true, color: C.text1, align: "right", valign: "middle", ...AR }, text: "" } },
      { text: { text: "TestPoint", options: { x: 0.6, y: 7.0, w: 2, h: 0.3, fontSize: 10, color: C.accent5, align: "left" } } },
    ],
    slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "7F8C8D", align: "right" },
  });

  const badge = (s, n) => {
    s.addShape(pres.shapes.OVAL, { x: 11.85, y: 0.35, w: 0.8, h: 0.8, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "Step " + n });
    s.addText(String(n), { x: 11.85, y: 0.35, w: 0.8, h: 0.8, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  const shot = (s, im, x, y, w, h, name) => {
    let iw = w, ih = w / im.r;
    if (ih > h) { ih = h; iw = h * im.r; }
    const ix = x + (w - iw) / 2, iy = y + (h - ih) / 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ix - 0.07, y: iy - 0.07, w: iw + 0.14, h: ih + 0.14, rectRadius: 0.06,
      fill: { color: C.background1 }, line: { color: "D5DDDA", width: 0.75 },
      shadow: { type: "outer", color: "000000", opacity: 0.18, blur: 8, offset: 3, angle: 90 }, objectName: name + " frame" });
    s.addImage({ data: im.data, x: ix, y: iy, w: iw, h: ih, altText: name, objectName: name });
    return { ix, iy, iw, ih };
  };
  const caption = (s, text, x, y, w) => s.addText(text, { x, y, w, h: 0.45, fontSize: 14, color: C.text2, align: "center", valign: "top", isTextBox: true, ...AR });
  const points = (s, items, x, y, w, h, size = 18) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize: size, color: C.text1, align: "right", valign: "top", paraSpaceAfter: 12, isTextBox: true, ...AR });
  const stat = (s, big, small, x, y, w, color) => {
    s.addText(big, { x, y, w, h: 0.9, fontSize: 46, bold: true, color: color || C.accent1, align: "center", valign: "bottom", margin: 0, isTextBox: true });
    s.addText(small, { x, y: y + 0.95, w, h: 0.7, fontSize: 14, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true, ...AR });
  };
  // two screens, one above the other (the screenshots are wide)
  const pair = (s, a, b, ca, cb) => {
    shot(s, a, 0.6, 1.35, 12.13, 2.55, ca); caption(s, ca, 0.6, 3.95, 12.13);
    shot(s, b, 0.6, 4.35, 12.13, 2.55, cb); caption(s, cb, 0.6, 6.92 - 0.02, 12.13);
  };
  const side = (s, a, b, ca, cb) => {
    shot(s, a, 6.85, 1.45, 5.9, 4.9, ca); caption(s, ca, 6.85, 6.45, 5.9);
    shot(s, b, 0.6, 1.45, 5.9, 4.9, cb); caption(s, cb, 0.6, 6.45, 5.9);
  };

  const D = n => path.join(SHOTS, n);
  const STATION = [0.38, 0.07, 0.62, 0.93];
  const im = {
    weigh: await img(D("01-weigh-in-face-detected.jpg"), [0.2, 0.08, 0.6, 0.79]),
    raise: await img(D("07-start-recognised-raise-hand.jpg"), [0.5, 0.08, 0.3, 0.88]),
    hand: await img(D("09-on-course-clock.jpg"), [0.5, 0.08, 0.3, 0.88]),
    finish: await img(D("23-finish-finishers.png"), [0, 0.38, 1, 0.62]),
    countdown: await img(D("17-station-countdown.jpg"), STATION),
    ready: await img(D("16-station-manual-start-hold.jpg"), STATION),
    crunchDown: await img(D("28-crunch-app-start.jpg"), [0.437, 0.085, 0.35, 0.75]),
    crunchUp: await img(D("29-crunch-app-up.jpg"), [0.437, 0.085, 0.35, 0.75]),
    coming: await img(D("27-finish-app-coming.jpg"), [0.21, 0.083, 0.287, 0.43]),
    timer: await img(D("30-pushup-count-20.jpg"), [0.533, 0.085, 0.252, 0.645]),

  };

  pres.addSection({ title: "TestPoint" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "TestPoint" });
  s.addText("TestPoint: اختبار اللياقة آلياً", { placeholder: "title" });
  s.addText("من التسجيل حتى النتيجة، بالكاميرات والتعرف على الوجه\nتجربة 2 نوفمبر 2026", { placeholder: "body" });
  s.addNotes("كل الصور من تجربة اليوم.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "TestPoint" });
  s.addText("التسجيل وخط البداية: الكاميرا تتعرف على المرشح", { placeholder: "title" }); badge(s, 1);
  side(s, im.weigh, im.raise, "التسجيل: اكتشاف الوجه عند الوصول", "خط البداية: التعرف عليه ثم طلب رفع اليد");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "TestPoint" });
  s.addText("الانطلاق والوصول: مؤقت خاص لكل مرشح ووقت آلي", { placeholder: "title" }); badge(s, 2);
  shot(s, im.hand, 8.65, 1.4, 4.1, 5.0, "Runner on the course with his own clock"); caption(s, "بعد الانطلاق: مؤقته يعمل «ما زال يجري»", 8.65, 6.5, 4.1);
  { const b = shot(s, im.coming, 0.6, 1.35, 7.7, 3.9, "Arriving at the finish line"); caption(s, "شاشة خط النهاية: الكاميرا تتبع العداء حتى يعبر الخط الأحمر", 0.6, b.iy + b.ih + 0.1, 7.7); }
  { const b = shot(s, im.finish, 0.6, 5.75, 7.7, 0.95, "Finishers list"); caption(s, "قائمة الواصلين: الوقت 1:05.5، تعرف بالوجه 70%", 0.6, b.iy + b.ih + 0.05, 7.7); }

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "TestPoint" });
  s.addText("محطة التمارين، الضغط: جاهز، ثم عدّ ووقت بخط كبير", { placeholder: "title" }); badge(s, 3);
  side(s, im.ready, im.timer, "1. جاهز في وضعية البداية «اثبت...»", "2. أثناء الاختبار: العدد 20 والوقت المتبقي 1:33");
  s.addText("الأرقام في هذه الصور من فيديو تجربة مكرر", { x: 0.6, y: 6.85, w: 12.13, h: 0.3, fontSize: 11, color: C.accent5, align: "center", isTextBox: true, ...AR });

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "TestPoint" });
  s.addText("محطة التمارين، البطن: حركتان تقيسهما الكاميرا", { placeholder: "title" }); badge(s, 4);
  side(s, im.crunchDown, im.crunchUp, "1. بداية الصعود من وضعية الاستلقاء", "2. الأعلى: الجذع يرتفع فيُحسب تمرين");
  s.addText("الأرقام في هذه الصور من فيديو تجربة اليوم", { x: 0.6, y: 6.85, w: 12.13, h: 0.3, fontSize: 11, color: C.accent5, align: "center", isTextBox: true, ...AR });

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "TestPoint" });
  s.addText("ما تحسّن اليوم، على تسجيلات حقيقية", { placeholder: "title" });
  const cards = [["10/10", "ضغط محسوب صحيحاً\n(كان 0 من 10)"], ["10/10", "بطن محسوب صحيحاً\n(كان 7 من 10)"], ["26", "إطار/ثانية مباشرة\n(كان 12)"], ["0", "عدّ خاطئ عند الجلوس أو الوقوف"]];
  const kw = 2.8, kg = (12.13 - 4 * kw) / 3;
  cards.forEach(([big, txt], i) => {
    const x = 12.73 - kw - i * (kw + kg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.8, w: kw, h: 3.3, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "Card " + (i + 1) });
    stat(s, big, txt, x + 0.1, 2.15, kw - 0.2);
  });
  s.addText("تم التحقق على تسجيل مرشح واحد بكاميرا واحدة؛ يلزم تسجيل 5-8 أشخاص قبل الاعتماد", { x: 0.6, y: 5.6, w: 12.13, h: 0.6, fontSize: 16, italic: true, color: C.accent4, align: "center", isTextBox: true, ...AR });

  s = pres.addSlide({ masterName: "DARK", sectionTitle: "TestPoint" });
  s.addText("الخطوات التالية", { placeholder: "title" });
  s.addText([
    { text: "تسجيل 5-8 أشخاص بعدات صحيحة وخاطئة لاعتماد العدّ", options: { bullet: true, breakLine: true } },
    { text: "تحديد معايير الاختبار الرسمية (عمق الضغط، ارتفاع البطن)", options: { bullet: true, breakLine: true } },
    { text: "تثبيت الكاميرات وإضاءة المكان قبل التجربة الكاملة", options: { bullet: true } },
  ], { placeholder: "body" });

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();













