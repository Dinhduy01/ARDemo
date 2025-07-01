package com.duy.nguyen.ardemo.helpers;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

public class ImageUtils {

    private static final String TAG = "ImageUtils";

    public static Bitmap imageToBitmap(Image image) {
        if (image == null) {
            Log.e(TAG, "Input image is null");
            return null;
        }

        try {
            Image.Plane[] planes = image.getPlanes();
            if (planes == null || planes.length < 3) {
                Log.e(TAG, "Invalid image planes: " + (planes == null ? "null" : planes.length));
                return null;
            }

            ByteBuffer yBuffer = planes[0].getBuffer(); // Y
            ByteBuffer uBuffer = planes[1].getBuffer(); // U
            ByteBuffer vBuffer = planes[2].getBuffer(); // V

            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();

            byte[] nv21 = new byte[ySize + uSize + vSize];

            // Copy Y channel
            yBuffer.get(nv21, 0, ySize);
            // Copy V and U channel - NOTE: NV21 format is Y + VU
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);

            YuvImage yuvImage = new YuvImage(
                    nv21,
                    ImageFormat.NV21,
                    image.getWidth(),
                    image.getHeight(),
                    null
            );

            ByteArrayOutputStream out = new ByteArrayOutputStream();
            boolean success = yuvImage.compressToJpeg(
                    new Rect(0, 0, image.getWidth(), image.getHeight()),
                    90,
                    out
            );

            if (!success) {
                Log.e(TAG, "compressToJpeg failed");
                return null;
            }

            byte[] jpegBytes = out.toByteArray();
            return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);

        } catch (Exception e) {
            Log.e(TAG, "Exception in imageToBitmap: " + e.getMessage(), e);
            return null;
        }
    }
}
