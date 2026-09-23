package com.drone.detector

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

internal object AssetPack {
    fun load(context: Context): ByteBuffer {
        val packed = context.assets.open(NAME).use { it.readBytes() }
        val iv = packed.copyOfRange(0, 12)
        val body = packed.copyOfRange(12, packed.size)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key(), "AES"), GCMParameterSpec(128, iv))
        val plain = cipher.doFinal(body)
        val buffer = ByteBuffer.allocateDirect(plain.size).order(ByteOrder.nativeOrder())
        buffer.put(plain)
        buffer.rewind()
        plain.fill(0)
        return buffer
    }

    private fun key(): ByteArray {
        val a = byteArrayOf(12, -106, 103, -28, 120, -55, 81, 90, 73, 121, -7, -16, -1, -43, -86, -90, 103, 49, -78, -82, 84, 40, 82, -120, -1, 99, -106, 118, -16, -11, 24, -82)
        val b = byteArrayOf(1, -42, 100, -53, -65, 99, 126, -98, -127, 24, 86, 26, 99, -78, -67, 90, 2, -25, -124, 103, 79, -31, 89, -5, -2, 108, -78, -93, -124, -125, 10, 91)
        return ByteArray(32) { i -> (a[i].toInt() xor b[i].toInt()).toByte() }
    }

    private const val NAME = "cfg.dat"
}
