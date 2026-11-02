# Management deck

`TestPoint-management-2026-11-02.pptx`: 7 slides in Arabic, screenshots from the trial on 2 November 2026.

To change it: edit `build-deck.js` (slides, captions) or the pictures in `screenshots/`, then

```bash
npm install pptxgenjs sharp
node build-deck.js
```

The push-up and crunch counts on the station slides come from the test videos replayed through the app (`test-videos/`), as the slides say.

The deck's colours are written by a helper (`apply_theme.js`) that is not in this repository; without it a rebuilt deck uses Office's default theme colours. Put the helper next to `build-deck.js` (or point `APPLY_THEME` at it) to get the same colours.
