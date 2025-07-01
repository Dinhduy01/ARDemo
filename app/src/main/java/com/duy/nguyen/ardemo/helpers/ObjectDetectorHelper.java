package com.duy.nguyen.ardemo.helpers;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.IOException;
import java.util.List;

public class ObjectDetectorHelper {

    private static final String TAG = "ObjectDetectorHelper";

    private float threshold = 0.5f;
    private int numThreads = 2;
    private int maxResults = 3;
    private int currentDelegate = DELEGATE_CPU;
    private int currentModel = MODEL_Sign4;

    private final Context context;
    private final DetectorListener objectDetectorListener;
    private ObjectDetector objectDetector;

    public ObjectDetectorHelper(
            float threshold,
            int numThreads,
            int maxResults,
            int currentDelegate,
            int currentModel,
            Context context,
            DetectorListener listener
    ) {
        this.threshold = threshold;
        this.numThreads = numThreads;
        this.maxResults = maxResults;
        this.currentDelegate = currentDelegate;
        this.currentModel = currentModel;
        this.context = context;
        this.objectDetectorListener = listener;

        setupObjectDetector();
    }

    public void clearObjectDetector() {
        objectDetector = null;
    }

    public void setupObjectDetector() {
        ObjectDetector.ObjectDetectorOptions.Builder optionsBuilder =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setScoreThreshold(threshold)
                        .setMaxResults(maxResults);

        BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads);

        switch (currentDelegate) {
            case DELEGATE_CPU:
                // Default
                break;
            case DELEGATE_GPU:
                if (new CompatibilityList().isDelegateSupportedOnThisDevice()) {
                    baseOptionsBuilder.useGpu();
                } else {
                    if (objectDetectorListener != null) {
                        objectDetectorListener.onError("GPU is not supported on this device");
                    }
                }
                break;
            case DELEGATE_NNAPI:
                baseOptionsBuilder.useNnapi();
                break;
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build());

        String modelName;
        switch (currentModel) {
            case MODEL_Sign4:
            default:
                modelName = "mobilenet1.tflite";
                break;
        }

        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(
                    context,
                    modelName,
                    optionsBuilder.build()
            );
        } catch (IllegalStateException | IOException e) {
            if (objectDetectorListener != null) {
                objectDetectorListener.onError("Object detector failed to initialize. See error logs for details.");
            }
            Log.e(TAG, "TFLite failed to load model with error: " + e.getMessage());
        }
    }

    public void detect(Bitmap image, int imageRotation) {
        if (objectDetector == null) {
            setupObjectDetector();
        }

        long inferenceTime = SystemClock.uptimeMillis();

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new Rot90Op(-imageRotation / 90))
                .build();

        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));
        List<Detection> results = objectDetector.detect(tensorImage);
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime;

        if (objectDetectorListener != null) {
            objectDetectorListener.onResults(results, inferenceTime, tensorImage.getHeight(), tensorImage.getWidth());
        }
    }

    public interface DetectorListener {
        void onError(String error);

        void onResults(List<Detection> results, long inferenceTime, int imageHeight, int imageWidth);
    }

    public static final int DELEGATE_CPU = 0;
    public static final int DELEGATE_GPU = 1;
    public static final int DELEGATE_NNAPI = 2;

    public static final int MODEL_Sign1 = 0;
    public static final int MODEL_Sign2 = 1;
    public static final int MODEL_Sign3 = 2;
    public static final int MODEL_Sign4 = 3;
}
