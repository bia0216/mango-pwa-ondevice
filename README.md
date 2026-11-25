# Mango Grader On-device (prototype 2)

A separate, experimental version of the Mango Grader PWA. The goal: the models run **inside the app on the phone**, so grading works without the Colab server.

This is **not** the version presented on 26 November 2026. That one stays at https://bia0216.github.io/mango-pwa/ and is not changed by this repo.

## Status

- Work in progress. Right now this repo has the same app as the presented prototype (v15), with separate storage, cache and app name, so both apps can be installed side by side.
- Next: add the YOLO models (mango detector, ripeness, disease) as ONNX files in `models/`, run them with ONNX Runtime Web, and check the results against the original models.
- Later: the Keras models (MobileNetV2, EfficientNetV2-B0), after a conversion test.
- The Colab server stays optional, for Grad-CAM heat maps and the Keras models.

## Live link

https://bia0216.github.io/mango-pwa-ondevice/ (after GitHub Pages is turned on)

Never upload API keys or data files to this repository.
