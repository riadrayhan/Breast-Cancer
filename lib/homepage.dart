import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_tflite/flutter_tflite.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:developer' as devtools;

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? filePath;
  String label = '';
  double confidence = 0.0;
  String message = '';
  final double confidenceThreshold = 0.7;

  Future<void> _tfLiteInit() async {
    try {
      String? res = await Tflite.loadModel(
          model: "assets/model_unquant.tflite",
          labels: "assets/labels.txt",
          numThreads: 1,
          isAsset: true,
          useGpuDelegate: false
      );
      devtools.log("Model loading result: $res");
    } catch (e) {
      devtools.log("Error loading model: $e");
      setState(() {
        message = 'Error: Unable to load model';
      });
    }
  }

  bool isValidImage(File image) {
    return image.lengthSync() > 0;
  }

  Future<void> processImage(File imageFile) async {
    setState(() {
      filePath = imageFile;
      message = 'Processing image...';
    });

    if (!isValidImage(imageFile)) {
      setState(() {
        message = 'Invalid image';
        label = '';
        confidence = 0.0;
      });
      return;
    }

    try {
      var recognitions = await Tflite.runModelOnImage(
          path: imageFile.path,
          imageMean: 0.0,
          imageStd: 255.0,
          numResults: 2,
          threshold: 0.1,
          asynch: true
      );

      devtools.log("Raw recognitions: $recognitions");

      if (recognitions == null || recognitions.isEmpty) {
        setState(() {
          message = 'No model data';
          label = '';
          confidence = 0.0;
        });
        return;
      }

      var topResult = recognitions[0];
      double resultConfidence = topResult['confidence'];

      setState(() {
        if (resultConfidence >= confidenceThreshold) {
          confidence = resultConfidence * 100;
          label = topResult['label'].toString();
          message = '';
        } else {
          message = 'Unable to classify with high confidence';
          label = '';
          confidence = 0.0;
        }
      });
    } catch (e) {
      devtools.log("Error running model: $e");
      setState(() {
        message = 'Error: Unable to process image';
      });
    }
  }

  Future<void> pickImageGallery() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image == null) return;

    await processImage(File(image.path));
  }

  Future<void> pickImageCamera() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.camera);

    if (image == null) return;

    await processImage(File(image.path));
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _tfLiteInit();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(" Breast Cancer"),
        centerTitle: true,
        backgroundColor: Colors.blueAccent[100],
        actions: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: IconButton(onPressed: () {
            }, icon: Icon(Icons.settings,color: Colors.white,)),
          )
        ],
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            children: [
              const SizedBox(height: 12),
              Card(
                elevation: 20,
                clipBehavior: Clip.hardEdge,
                child: SizedBox(
                  width: 300,
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        const SizedBox(height: 18),
                        Container(
                          height: 280,
                          width: 280,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(12),
                            image: const DecorationImage(
                              image: AssetImage('assets/upload.jpg'),
                            ),
                          ),
                          child: filePath == null
                              ? const Text('')
                              : Image.file(
                            filePath!,
                            fit: BoxFit.fill,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Column(
                            children: [
                              if (message.isNotEmpty)
                                Text(
                                  message,
                                  style: const TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.red,
                                  ),
                                )
                              else ...[
                                Text(
                                  label,
                                  style: const TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 12),
                                Text(
                                  "The Accuracy is ${confidence.toStringAsFixed(0)}%",
                                  style: const TextStyle(
                                    fontSize: 18,
                                  ),
                                ),
                                const SizedBox(height: 12),
                              ],
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: pickImageCamera,
                style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 30, vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(13),
                    ),
                    foregroundColor: Colors.black),
                child: const Text(
                  "Take a Photo",
                ),
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: pickImageGallery,
                style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 30, vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(13),
                    ),
                    foregroundColor: Colors.black),
                child: const Text(
                  "Pick from gallery",
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}