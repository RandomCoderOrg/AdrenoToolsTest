#pragma once

#include <android/native_window.h>

bool panvk_buffer_copy_probe(void *libvulkan, const char *label);
bool panvk_present_init(void *libvulkan, ANativeWindow *window);
void panvk_present_draw();
void panvk_present_shutdown();
