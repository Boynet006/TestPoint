# Management deck

`TestPoint-management-2026-11-02.pptx`: 7 slides in Arabic, screenshots from the trial on 2 November 2026.

To change it: edit `build-deck.js` (slides, captions) or the pictures in `screenshots/`, then

```bash
npm install pptxgenjs sharp
node build-deck.js
```

The push-up and crunch counts on the station slides come from the test videos replayed through the app (`test-videos/`), as the slides say.

The deck's colours are written by a helper (`apply_theme.js`) that is not in this repository; without it a rebuilt deck uses Office's default theme colours. Put the helper next to `build-deck.js` (or point `APPLY_THEME` at it) to get the same colours.

## Walk-through deck

`TestPoint-walkthrough-2026-11-03.pptx`: 22 slides in Arabic, one person through the whole test day (weigh-in, start, finish, push-ups, crunches, results, the Advanced screens, settings), each screenshot the whole app window uncropped (`walkthrough-2026-11-03/`). Made on a test copy of the app with its own database, so no real results were touched. It ends with the bugs that the walk-through found and fixed.

```bash
node build-walkthrough.js walkthrough-2026-11-03 <pptx skill folder> TestPoint-walkthrough-2026-11-03.pptx
```
