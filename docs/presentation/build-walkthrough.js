// TestPoint: a whole test day, screen by screen (Arabic, RTL). Every screenshot is the whole app
// window, uncropped. Usage: node walk-deck.js <shots dir> <pptx skill dir> <out.pptx>
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
const { applyTheme } = require(path.join(process.argv[3], "scripts", "apply_theme.js"));
const SHOTS = process.argv[2], OUT = process.argv[4];

// The app's own Desert Stars colours: night sand, amber, green and red.
const THEME = { name: "TestPoint Desert", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: { dk1: "16120D", lt1: "F1E8D7", dk2: "221C15", lt2: "B2A283",
    accent1: "D4A24C", accent2: "3BA866", accent3: "D2353F", accent4: "3B3226", accent5: "7A6B53", accent6: "E9C77B",
    hlink: "D4A24C", folHlink: "B2A283" } };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5
pres.rtlMode = true;
pres.title = "يوم اختبار كامل في TestPoint";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const AR = { rtlMode: true, lang: "ar-AE", align: "right" };

pres.defineSlideMaster({ title: "Title", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.73, h: 1.5, fontSize: 46, bold: true, color: C.background1, align: "right", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.7, w: 11.73, h: 2.4, fontSize: 22, color: C.background2, align: "right", valign: "top", margin: 0 }, text: "" } },
  ] });
pres.defineSlideMaster({ title: "Screen", background: { color: C.text1 },
  slideNumber: { x: 0.45, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.background2 },
  objects: [
    { text: { text: "TestPoint", options: { x: 11.0, y: 7.05, w: 1.88, h: 0.3, fontSize: 10, color: C.background2, align: "right", margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 1.6, y: 0.28, w: 10.4, h: 0.62, fontSize: 28, bold: true, color: C.background1, align: "right", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.45, y: 0.9, w: 11.55, h: 0.42, fontSize: 15, color: C.background2, align: "right", valign: "top", margin: 0 }, text: "" } },
  ] });
pres.defineSlideMaster({ title: "Content", background: { color: C.text1 },
  slideNumber: { x: 0.45, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.background2 },
  objects: [
    { text: { text: "TestPoint", options: { x: 11.0, y: 7.05, w: 1.88, h: 0.3, fontSize: 10, color: C.background2, align: "right", margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.35, w: 12.43, h: 0.8, fontSize: 32, bold: true, color: C.background1, align: "right", valign: "middle", margin: 0 }, text: "" } },
  ] });

// a numbered step badge, the same on every screen slide
const badge = (s, n) => {
  s.addShape(pres.shapes.OVAL, { objectName: "Step badge", x: 12.25, y: 0.3, w: 0.62, h: 0.62, fill: { color: C.accent1 } });
  s.addText(String(n), { objectName: "Step number", isTextBox: true, x: 12.25, y: 0.3, w: 0.62, h: 0.62, fontSize: 20, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0 });
};
// the whole app window, uncropped (1544 x 784 or 1512 x 812), as large as the slide allows
const SIZE = {};   // every shot is the whole window, 1544 x 784
function screen(s, file, alt) {
  const key = file.slice(0, 2), [pw, ph] = SIZE[key] || [1544, 784], r = pw / ph;
  const maxW = 12.43, maxH = 5.55, top = 1.42;
  let w = maxW, h = w / r; if (h > maxH) { h = maxH; w = h * r; }
  const x = (13.33 - w) / 2;
  s.addShape(pres.shapes.RECTANGLE, { objectName: "Screen frame", x: x - 0.04, y: top - 0.04, w: w + 0.08, h: h + 0.08, fill: { color: C.accent4 }, line: { color: C.accent4 } });
  s.addImage({ objectName: "Screenshot", path: path.join(SHOTS, file), x, y: top, w, h, altText: alt });
}
let n = 0;
function screenSlide(section, title, sub, file) {
  const s = pres.addSlide({ masterName: "Screen", sectionTitle: section });
  s.addText(title, { placeholder: "title", ...AR });
  s.addText(sub, { placeholder: "body", ...AR });
  badge(s, ++n);
  screen(s, file, `${title}: ${sub}`);
  return s;
}

// ---- 1. title
pres.addSection({ title: "البداية" });
let s = pres.addSlide({ masterName: "Title", sectionTitle: "البداية" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: "Logo", x: 11.73, y: 1.05, w: 0.8, h: 0.8, rectRadius: 0.08, fill: { color: C.accent1 } });
s.addText("T", { objectName: "Logo letter", isTextBox: true, x: 11.73, y: 1.05, w: 0.8, h: 0.8, fontSize: 30, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0 });
s.addText("يوم اختبار كامل في TestPoint", { placeholder: "title", ...AR });
s.addText([{ text: "جولة على كل الشاشات، من الوزن حتى النتيجة", options: { breakLine: true } },
  { text: "3 نوفمبر 2026 · كل صورة هي نافذة التطبيق كاملة", options: { color: C.accent1, fontSize: 18 } }], { placeholder: "body", ...AR, paraSpaceAfter: 8 });
s.addNotes("جولة كاملة على نسخة تجريبية من التطبيق بقاعدة بيانات منفصلة: مرشح واحد (عمر) من الوزن حتى النتيجة، مع مرشحَين آخرين في القوائم.");

// ---- 2. the day in six steps
s = pres.addSlide({ masterName: "Content", sectionTitle: "البداية" });
s.addText("يوم الاختبار في ست محطات", { placeholder: "title", ...AR });
const steps = [["الوزن", "الوجه والوزن ومؤشر الكتلة"], ["خط البداية", "عند الخط، يرفع يده، انطلق"], ["خط النهاية", "الكاميرا ترى العبور"],
  ["الضغط", "عدّ آلي لدقيقتين"], ["البطن", "على البساط نفسه"], ["النتيجة", "ناجح أو راسب، لكل اختبار"]];
const bw = 1.82, gap = 0.27, by = 2.55;
steps.forEach(([t, d], i) => {
  const x = 12.88 - (i + 1) * bw - i * gap;     // right to left
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: `Step ${i + 1}`, x, y: by, w: bw, h: 2.5, rectRadius: 0.1, fill: { color: C.text2 }, line: { color: i === 5 ? C.accent2 : C.accent4, width: 1 } });
  s.addShape(pres.shapes.OVAL, { objectName: `Step ${i + 1} badge`, x: x + bw / 2 - 0.32, y: by + 0.28, w: 0.64, h: 0.64, fill: { color: i === 5 ? C.accent2 : C.accent1 } });
  s.addText(String(i + 1), { objectName: `Step ${i + 1} number`, isTextBox: true, x: x + bw / 2 - 0.32, y: by + 0.28, w: 0.64, h: 0.64, fontSize: 20, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0 });
  s.addText(t, { objectName: `Step ${i + 1} name`, isTextBox: true, x: x + 0.1, y: by + 1.05, w: bw - 0.2, h: 0.5, fontSize: 20, bold: true, color: C.background1, align: "center", margin: 0, rtlMode: true, lang: "ar-AE" });
  s.addText(d, { objectName: `Step ${i + 1} note`, isTextBox: true, x: x + 0.12, y: by + 1.6, w: bw - 0.24, h: 0.75, fontSize: 14, color: C.background2, align: "center", valign: "top", margin: 0, rtlMode: true, lang: "ar-AE" });
  if (i < 5) s.addText("←", { objectName: `Arrow ${i + 1}`, isTextBox: true, x: x - gap - 0.02, y: by + 1.0, w: gap + 0.04, h: 0.5, fontSize: 18, color: C.accent1, align: "center", margin: 0 });
});
s.addText("كل محطة تعرف المرشح من وجهه، والفاحص يتدخل عند الحاجة فقط. كل شيء مسجّل بالفيديو ويمكن مراجعته لاحقاً.",
  { objectName: "Summary", isTextBox: true, x: 0.45, y: 5.55, w: 12.43, h: 0.8, fontSize: 18, color: C.background1, ...AR, margin: 0 });

// ---- 3. the screens
pres.addSection({ title: "الوزن والجري" });
screenSlide("الوزن والجري", "الوزن", "تعرّف على عمر من صورته (تطابق 0.96)، ثم الوزن والطول ومؤشر الكتلة، وما يحتاجه للنجاح حسب عمره", "01-weigh-in.jpg");
screenSlide("الوزن والجري", "خط البداية", "الكاميرا تتعرّف عليه عند الخط وتطلب منه رفع يده للانطلاق", "02-start-line.jpg");
screenSlide("الوزن والجري", "الانطلاق برفع اليد", "أي يد مرفوعة تُطلق الانطلاق (الإطار الأخضر «HAND UP»)؛ هنا مع عدّ 3-2-1، والافتراضي الآن الانطلاق فوراً", "03-start-countdown.jpg");
screenSlide("الوزن والجري", "في المسار", "لكل عدّاء مؤقته الخاص من لحظة انطلاقه", "04-on-course.jpg");
screenSlide("الوزن والجري", "خط النهاية: الكاميرا", "فيديو التجربة بسرعته الحقيقية مع أزرار التشغيل؛ الخط الأحمر هو خط النهاية", "05-finish-camera.jpg");
screenSlide("الوزن والجري", "الواصلون", "الوقت 1:07.8 والنتيجة 100% مع مقطع فيديو للعبور", "06-finishers.jpg");

pres.addSection({ title: "محطة التمارين" });
screenSlide("محطة التمارين", "اختيار الاختبار الأول", "صورة الكاميرا كمرآة على الشاشة كلها؛ يرفع يده إلى أحد الجانبين ويثبتها ثانية ونصفاً", "07-choose-first-test.jpg");
screenSlide("محطة التمارين", "الضغط", "الكاميرا تعدّ: 9 تكرارات والوقت المتبقي 1:47", "08-pushups-counting.jpg");
screenSlide("محطة التمارين", "البطن", "على البساط نفسه بعد الضغط", "09-crunches-counting.jpg");
screenSlide("محطة التمارين", "نتيجة المحطة", "انتهى الاختبار عندما توقف عن التكرار؛ العدد حتى الآن هو نتيجته، ويمكن إعادة الاختبار", "10-station-result.jpg");
screenSlide("محطة التمارين", "شاشة النتائج", "الجري والضغط والبطن معاً، والنتيجة النهائية", "11-results.jpg");

pres.addSection({ title: "المتابعة والمراجعة" });
screenSlide("المتابعة والمراجعة", "قائمة اليوم", "كل المرشحين وحالاتهم ونتائجهم؛ التواريخ بصيغة يوم-شهر-سنة", "12-todays-list.jpg");
screenSlide("المتابعة والمراجعة", "المرشحون", "السجل الدائم: العمر والفئة وآخر اختبار", "13-candidates.jpg");
screenSlide("المتابعة والمراجعة", "صور الوجه", "كل صور الشخص: تنزيل صورة أو الكل، حذف صورة رديئة، أو فتحها في المستكشف", "14-face-photos.jpg");
screenSlide("المتابعة والمراجعة", "المراجعة", "العدد المحفوظ بجانب عدّ الكاميرا، وكل تكرار بمقطعه", "15-review.jpg");
screenSlide("المتابعة والمراجعة", "التسجيلات", "أي يوم وأي مرشح: الانطلاق والوصول وفيديو المحاولة كاملاً وكل تكرار", "16-recordings.jpg");

pres.addSection({ title: "الإعدادات" });
screenSlide("الإعدادات", "الإعدادات: الكاميرات", "لكل شاشة مصادر الكاميرا الخاصة بها، وعناوين كاميرات IP", "17-settings-cameras.jpg");
screenSlide("الإعدادات", "الإعدادات: انطلاق الجري", "العد التنازلي 0 ث: الانطلاق والعدّ يبدآن فور استعداد الشخص", "18-settings-run-start.jpg");
screenSlide("الإعدادات", "الإعدادات: نهاية الجري", "تطابق الوجه عند النهاية، وإظهار إعداد خط النهاية أو إخفاؤه", "19-settings-run-finish.jpg");
screenSlide("الإعدادات", "الإعدادات: محطة التمارين", "متى ينتهي الاختبار، واختيار الاختبار الأول، وقواعد العدّ", "20-settings-station.jpg");
screenSlide("الإعدادات", "القائمة", "اللغة والعرض وملء الشاشة وملاءمة النافذة والإعدادات في قائمة واحدة", "21-menu.jpg");
screenSlide("الإعدادات", "بالعربية", "الواجهة كلها بالعربية ومن اليمين إلى اليسار، والتاريخ والوقت بالصيغة نفسها", "22-finish-arabic.jpg");

// ---- what this walk-through fixed
pres.addSection({ title: "الخلاصة" });
s = pres.addSlide({ masterName: "Content", sectionTitle: "الخلاصة" });
s.addText("ما أُصلح خلال هذه الجولة", { placeholder: "title", ...AR });
const fixes = [
  ["خط البداية", "رفع اليد يُرى حتى دون ظهور الوركين، ولا «ما زال يجري» بالأحمر بعد ضغط الفاحص «بدء»"],
  ["الانطلاق", "بلا عدّ تنازلي: الوقت والعدّ يبدآن عند الاستعداد (قابل للتغيير في الإعدادات)"],
  ["المحطة", "اليد على الذقن لا تختار اختباراً، ووضعية البداية تختاره، ولا اختيار مع فيديو تجريبي"],
  ["الكاميرات", "إطار الوجه في مكانه مع كاميرا الخدمة، ولا تعلّم للوجه من صورة عليها رسوم"],
  ["صور الوجه", "عارض لكل صور الوجه: تنزيل وحذف وفتح في المستكشف"],
  ["التاريخ والوقت", "يوم-شهر-سنة وساعة 24 في كل الشاشات وباللغتين؛ التاريخ لا ينقلب بالعربية"],
  ["اللغة", "تبديل اللغة يعيد رسم الأزرار والعنوان الفرعي"],
  ["الرسائل والشاشات", "كلمات لكل رموز الأخطاء، ورسائل «جارٍ البحث» و«الكاميرا تبدأ» تختفي في وقتها، والصورة لا تقفز"],
];
const colW = 6.0, rowH = 1.05;
fixes.forEach(([tag, text], i) => {
  const col = i % 2, row = Math.floor(i / 2), x = 12.88 - (col + 1) * colW - col * 0.43, y = 1.45 + row * (rowH + 0.22);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { objectName: `Fix ${i + 1}`, x, y, w: colW, h: rowH, rectRadius: 0.08, fill: { color: C.text2 }, line: { color: C.accent4, width: 0.75 } });
  s.addText(tag, { objectName: `Fix ${i + 1} area`, isTextBox: true, x: x + colW - 1.55, y: y + 0.12, w: 1.4, h: 0.36, fontSize: 13, bold: true, color: C.accent1, ...AR, margin: 0 });
  s.addText(text, { objectName: `Fix ${i + 1} text`, isTextBox: true, x: x + 0.2, y: y + 0.42, w: colW - 0.35, h: rowH - 0.5, fontSize: 14, color: C.background1, ...AR, valign: "top", margin: 0 });
});
s.addNotes("كلها مختبرة: 282 اختباراً آلياً ناجحاً.");

// ---- next
s = pres.addSlide({ masterName: "Title", sectionTitle: "الخلاصة" });
s.addShape(pres.shapes.OVAL, { objectName: "Next badge", x: 11.73, y: 1.05, w: 0.8, h: 0.8, fill: { color: C.accent2 } });
s.addText("←", { objectName: "Next arrow", isTextBox: true, x: 11.73, y: 1.05, w: 0.8, h: 0.8, fontSize: 28, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0 });
s.addText("الخطوات التالية", { placeholder: "title", ...AR });
s.addText([
  { text: "تجربة كاملة بعدة مرشحين حقيقيين وكاميرات الموقع", options: { bullet: true, breakLine: true } },
  { text: "اعتماد قواعد العدّ مع الفاحصين (العمق، الراحة، متى ينتهي الاختبار)", options: { bullet: true, breakLine: true } },
  { text: "إخفاء مصادر التجربة (ملف فيديو، صورة) يوم الاختبار من الإعدادات", options: { bullet: true, breakLine: true } },
  { text: "حذف صورتَي الوجه اللتين عليهما خط (من عارض صور الوجه)", options: { bullet: true } },
], { placeholder: "body", ...AR, paraSpaceAfter: 10 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
