// File: include/watermark.h
#ifndef WATERMARK_H
#define WATERMARK_H

__global__ void apply_watermark(unsigned char *image, unsigned char *watermark, int width, int height);

#endif
