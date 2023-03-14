package com.auf.cea.tensorflowobjectdetection

import android.graphics.*
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.auf.cea.tensorflowobjectdetection.databinding.ActivityMainBinding
import com.otaliastudios.cameraview.frame.Frame
import com.otaliastudios.cameraview.frame.FrameProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity(), FrameProcessor {
    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.camera.setLifecycleOwner(this)
        binding.camera.addFrameProcessor(this)

    }

    override fun process(frame: Frame) {
        processDataFrame(frame)
    }

    private fun processDataFrame(frame: Frame){
        val bitmap = convertToBitmap(frame)
        if(bitmap != null){

            val tensorImage = TensorImage.fromBitmap(bitmap)

            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(2)
                .setScoreThreshold(0.5f)
                .build()

            val detector = ObjectDetector.createFromFileAndOptions(
                this,
                "lite-model.tflite",
                options
            )

            val results = detector.detect(tensorImage)
            //debugPrint(results)

            val resultsToDisplay = results.map {
                val category = it.categories.first()
                val text = "${category.label}, ${category.score.times(100).toInt()}%"

                DetectionResult(it.boundingBox,text)
            }

            val transBitmap = Bitmap.createBitmap(bitmap.width,bitmap.height,Bitmap.Config.ARGB_8888)
            val imgRes = drawDetectionResult(transBitmap,resultsToDisplay)
            runOnUiThread{
                binding.imgBoundingBox.setImageBitmap(imgRes)
            }

        }
    }

    private fun debugPrint(results : List<Detection>) {
        for ((i, obj) in results.withIndex()) {
            val box = obj.boundingBox

            Log.d(MainActivity::class.java.simpleName, "Detected object: ${i} ")
            Log.d(MainActivity::class.java.simpleName, "  boundingBox: (${box.left}, ${box.top}) - (${box.right},${box.bottom})")

            for ((j, category) in obj.categories.withIndex()) {
                Log.d(MainActivity::class.java.simpleName, "    Label $j: ${category.label}")
                val confidence: Int = category.score.times(100).toInt()
                Log.d(MainActivity::class.java.simpleName, "    Confidence: ${confidence}%")
            }
        }
    }

    /**
     * Draw bounding boxes around objects together with the object's name.
     */
    private fun drawDetectionResult(bitmap: Bitmap, detectionResults: List<DetectionResult>): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT

        detectionResults.forEach {
            // draw bounding box
            pen.color = Color.RED
            pen.strokeWidth = 8F
            pen.style = Paint.Style.STROKE
            val box = it.boundingBox
            canvas.drawRect(box, pen)


            val tagSize = Rect(0, 0, 0, 0)

            // calculate the right font size
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.WHITE
            pen.strokeWidth = 2F

            pen.textSize = 96F
            pen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()

            // adjust the font size so texts are inside the bounding box
            if (fontSize < pen.textSize) pen.textSize = fontSize

            var margin = (box.width() - tagSize.width()) / 2.0F
            if (margin < 0F) margin = 0F
            canvas.drawText(
                it.text, box.left + margin,
                box.top + tagSize.height().times(1F), pen
            )
        }
        return outputBitmap
    }


    private fun convertToBitmap(frame: Frame):Bitmap?{
        if(frame.dataClass == ByteArray::class.java){
            val out = ByteArrayOutputStream()
            val yuvImage = YuvImage(
                frame.getData(),
                ImageFormat.NV21,
                frame.size.width,
                frame.size.height,
                null
            )

            yuvImage.compressToJpeg(
                Rect(0,0,frame.size.width,frame.size.height),
                90,
                out
            )
            val imageByte = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageByte,0,imageByte.size)
        }else if(frame.dataClass == Image::class.java){
            val data = frame.getData() as Image
            val imageBuffer = data.planes[0].buffer
            val bytes = imageBuffer.array()
            return BitmapFactory.decodeByteArray(bytes,0,bytes.size)
        }

        return null
    }
}

/**
 * DetectionResult
 *      A class to store the visualization info of a detected object.
 */
data class DetectionResult(val boundingBox: RectF, val text: String)