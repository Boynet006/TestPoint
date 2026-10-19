# Demo on one computer

Until the cameras are installed, the whole system runs on one PC, with photos
and video files standing in for the cameras. Moving to real hardware later is
a choice in the camera-source picker, not a code change.

## Prepare (once)

1. Double-click **Start TestPoint.cmd**. The station opens at
   http://localhost:8080.
2. Switch to **Advanced → Today's list** and import the candidate list
   (`samples/candidates.csv` is an example; columns
   `external_id,name,name_ar,gender,dob,bib`).
3. For each demo candidate, press **Enrol photo** in the Face column and pick
   a clear photo of one face (passport-style is best). The thumbnail shows the
   aligned face that was saved. Up to 3 photos per person are kept.

Use photos of people who agreed to take part. Face data is personal data;
remove it after the demo with the ✕ next to the thumbnail.

## Run the demo

1. Back in **Simple**, choose **Weigh-in** as this device's role.
2. Choose the camera source:
   - **Photo**: pick a different photo of an enrolled person, then press
     **Identify face**.
   - **Video file**: pick a short clip of a person facing the camera and turn
     **Auto** on; the system identifies them while the video plays.
   - **Webcam**: plug in a USB webcam (Logitech C920/Brio or similar); works
     on this PC at `localhost` without extra setup.
3. When the face is recognised (green box, name and match score), the weigh-in
   form opens: type weight and height, then Accept or Reject.
4. If the face is not recognised ("not sure"), type the bib instead. If that
   candidate has no face on file yet, **Save this face for the candidate**
   stores the face on screen for next time.
5. Continue the day on the other roles from the same PC: **Run · start line**
   (mark present, START twice), **Run · finish line** (type bibs as they
   cross), **Exercise station** (check in per lane, tap-count, save).

## How recognition decides

- A match must reach a similarity of 0.363 (the SFace model's published
  threshold) **and** beat the second-best candidate by 0.05. Otherwise the
  screen says "not sure" and the examiner types the bib: a wrong person
  checked in is worse than five seconds of typing.
- Only people on **today's** list are searched.
- Enrolment is refused if the face already belongs to another candidate on
  file, so a mixed-up photo cannot create two people with one face.

## Limits of the demo

- Matching was checked on altered copies of the same photo (darker, rotated,
  blurred, resized: 12 of 12 correct, strangers all "not sure"). Two genuinely
  different photos of the same person, under real lighting, are harder; the
  real accuracy has to be measured with the webcam and real candidates.
- Faces smaller than 40 pixels are refused; at a check-in desk faces are
  several hundred pixels wide.
- A video or photo can be shown to the camera instead of a real person. Real
  check-in at the desk is supervised by the examiner; liveness detection is a
  later hardening item.
