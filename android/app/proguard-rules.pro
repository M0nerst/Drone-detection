-optimizationpasses 5
-allowaccessmodification
-repackageclasses ''
-dontusemixedcaseclassnames
-verbose

-keepattributes Signature
-keepattributes *Annotation*
-keepattributes InnerClasses
-keepattributes EnclosingMethod
-keepattributes SourceFile,LineNumberTable

-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }
-dontwarn org.tensorflow.lite.**
-dontwarn org.tensorflow.lite.gpu.**

-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

-keep class com.drone.detector.databinding.** { *; }
-keepclassmembers class * implements androidx.viewbinding.ViewBinding {
    public static *** bind(android.view.View);
    public static *** inflate(android.view.LayoutInflater);
}

-keep class kotlin.Metadata { *; }
-dontwarn javax.annotation.**
