package com.drone.detector

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

fun ImageProxy.rgbaToBitmap(): Bitmap {
    val plane = planes[0]
    val buffer = plane.buffer.duplicate()
    buffer.rewind()
    val pixelStride = plane.pixelStride.coerceAtLeast(1)
    val rowStride = plane.rowStride
    val rowPadding = rowStride - pixelStride * width
    val extra = if (pixelStride == 0) 0 else rowPadding / pixelStride
    val bitmap = Bitmap.createBitmap(width + extra, height, Bitmap.Config.ARGB_8888)
    bitmap.copyPixelsFromBuffer(buffer)
    val cropped = if (bitmap.width == width) {
        bitmap
    } else {
        val slice = Bitmap.createBitmap(bitmap, 0, 0, width, height)
        bitmap.recycle()
        slice
    }
    val rotation = imageInfo.rotationDegrees
    if (rotation == 0) return cropped
    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    val rotated = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
    if (rotated != cropped) cropped.recycle()
    return rotated
}
