package com.drone.detector

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.drone.detector.databinding.ActivityMainBinding
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var detector: YoloDetector? = null
    private var camera: Camera? = null
    private var currentZoom = 1f
    private var maxZoom = 1f
    private var minZoom = 1f
    private var busy = AtomicBoolean(false)
    private var lastUiTs = 0L
    private var frames = 0
    private var permissionAsked = false
    private lateinit var scaleDetector: ScaleGestureDetector

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            hidePermissionPanel()
            startCamera()
        } else {
            showPermissionDenied()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    applyZoom(currentZoom * detector.scaleFactor)
                    syncZoomUi()
                    return true
                }
            },
        )

        binding.previewView.setOnTouchListener { _, event ->
            scaleDetector.onTouchEvent(event)
            true
        }

        binding.zoomSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser || maxZoom <= minZoom) return
                val ratio = minZoom + (maxZoom - minZoom) * progress / seekBar!!.max
                applyZoom(ratio)
                binding.zoomValueText.text = "${"%.1f".format(currentZoom)}x"
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })

        binding.closeButton.setOnClickListener { finishAffinity() }
        binding.permissionButton.setOnClickListener {
            if (shouldOpenSettings()) {
                startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data = Uri.fromParts("package", packageName, null)
                })
            } else {
                permissionAsked = true
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        analysisExecutor.execute {
            try {
                detector = YoloDetector(this)
                runOnUiThread {
                    binding.statusText.text =
                        getString(R.string.model_ready) + "  |  INT8  |  ${detector?.backend}"
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Model load failed", t)
                runOnUiThread {
                    binding.statusText.text = getString(R.string.model_error)
                    Toast.makeText(this, t.message ?: "model", Toast.LENGTH_LONG).show()
                }
            }
        }

        if (hasCameraPermission()) {
            startCamera()
        } else {
            showPermissionRationale()
            permissionAsked = true
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = binding.previewView.surfaceProvider
            }
            val resolution = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 640), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setResolutionSelector(resolution)
                .build()
            analysis.setAnalyzer(analysisExecutor) { image -> analyze(image) }
            provider.unbindAll()
            camera = provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis,
            )
            observeZoom()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun observeZoom() {
        val cam = camera ?: return
        cam.cameraInfo.zoomState.observe(this) { state ->
            minZoom = state.minZoomRatio
            maxZoom = state.maxZoomRatio
            currentZoom = state.zoomRatio.coerceIn(minZoom, maxZoom)
            syncZoomUi()
        }
    }

    private fun applyZoom(ratio: Float) {
        val cam = camera ?: return
        if (maxZoom <= minZoom) return
        currentZoom = ratio.coerceIn(minZoom, maxZoom)
        cam.cameraControl.setZoomRatio(currentZoom)
    }

    private fun syncZoomUi() {
        if (maxZoom <= minZoom) return
        val progress = ((currentZoom - minZoom) / (maxZoom - minZoom) * binding.zoomSeekBar.max).toInt()
        binding.zoomSeekBar.progress = progress
        binding.zoomValueText.text = "${"%.1f".format(currentZoom)}x"
    }

    private fun analyze(image: androidx.camera.core.ImageProxy) {
        val model = detector
        if (model == null || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            val bitmap = image.rgbaToBitmap()
            try {
                val result = model.detect(bitmap)
                val viewBoxes = mapFillCenter(
                    result.detections,
                    bitmap.width.toFloat(),
                    bitmap.height.toFloat(),
                    binding.overlayView.width.toFloat(),
                    binding.overlayView.height.toFloat(),
                )
                frames += 1
                val now = System.currentTimeMillis()
                if (now - lastUiTs >= 250L) {
                    val fps = if (lastUiTs == 0L) 0f else frames * 1000f / (now - lastUiTs)
                    frames = 0
                    lastUiTs = now
                    runOnUiThread {
                        binding.statusText.text =
                            "Дронов: ${viewBoxes.size}  |  max ${(result.maxScore * 100).toInt()}%  |  ${"%.0f".format(fps)} FPS  |  ${result.backend}  ${result.inferMs}ms"
                    }
                }
                runOnUiThread { binding.overlayView.detections = viewBoxes }
            } finally {
                bitmap.recycle()
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Inference failed", t)
            runOnUiThread {
                binding.statusText.text = "Ошибка: ${t.javaClass.simpleName}"
            }
        } finally {
            busy.set(false)
            image.close()
        }
    }

    private fun mapFillCenter(
        detections: List<Detection>,
        imageWidth: Float,
        imageHeight: Float,
        viewWidth: Float,
        viewHeight: Float,
    ): List<Detection> {
        if (viewWidth <= 1f || viewHeight <= 1f || imageWidth <= 1f || imageHeight <= 1f) {
            return detections
        }
        val scale = max(viewWidth / imageWidth, viewHeight / imageHeight)
        val dx = (viewWidth - imageWidth * scale) / 2f
        val dy = (viewHeight - imageHeight * scale) / 2f
        return detections.map { det ->
            det.copy(
                left = det.left * scale + dx,
                top = det.top * scale + dy,
                right = det.right * scale + dx,
                bottom = det.bottom * scale + dy,
            )
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun shouldOpenSettings(): Boolean {
        return permissionAsked &&
            !hasCameraPermission() &&
            !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
    }

    private fun showPermissionRationale() {
        binding.permissionPanel.visibility = View.VISIBLE
        binding.permissionText.setText(R.string.camera_rationale)
        binding.permissionButton.setText(R.string.grant_camera)
    }

    private fun showPermissionDenied() {
        binding.permissionPanel.visibility = View.VISIBLE
        binding.permissionText.setText(R.string.camera_denied)
        binding.permissionButton.setText(R.string.open_settings)
    }

    private fun hidePermissionPanel() {
        binding.permissionPanel.visibility = View.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        detector?.close()
    }

    companion object {
        private const val TAG = "AntiUav"
    }
}
