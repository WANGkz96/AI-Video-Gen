# Vast template setup

Vast is the manual fallback for Video-pipeline's `live_vast` mode. The current
backend is ComfyUI LTX 2.5; old `MODELS=ltx-2.3-distilled`,
`GENERATOR_BACKEND=comfyui-ltx23`, and `AI_VIDEO_GEN_FORCE_LTX23_DISTILLED=1`
template variables are ignored by the current on-start script.

Use an Ubuntu GPU image with root or passwordless `sudo`, NVIDIA drivers,
`apt-get`, SSH, and a direct port for `8090`. Configure a 400 GB disk in
Video-pipeline. Set the on-start command to the image's `entrypoint.sh` and
point `PROVISIONING_SCRIPT` at:

```text
https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh
```

The script checks out AI-Video-Gen `master` and installs the pinned ComfyUI
LTX 2.5 workflows and LongCat runtime when required by the batch. The API
listens on port `8090`; ComfyUI stays private on port `18188`. Set
`OPEN_BUTTON_PORT=8090` if the Vast portal should open the API frontend.

See [Vast AI LTX 2.5 + LongCat template](vast-comfyui-template.md) for the
runtime variables, readiness checks, and model setup. No instance is started
by reading or applying these settings; `live_vast` requires an explicit run.
