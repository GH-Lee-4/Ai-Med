import React, { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { UploadCloud, Scan, Camera, Download } from "lucide-react";
import { motion } from "framer-motion";
import { useCamera } from "../hooks/useCamera";
import { useImageProcessing } from "../hooks/useImageProcessing";

export default function ImageProcessingUI() {
    const [image, setImage] = useState(null);
    const [tone, setTone] = useState("original");
    const { videoRef, canvasRef, streaming, startCamera, stopCamera, captureImage } = useCamera();
    const { handleImageUpload, downloadImage } = useImageProcessing(image, setImage);

    useEffect(() => {
        return () => stopCamera();
    }, []);

    return (
        <div className="flex flex-col items-center p-6 bg-gray-900 text-white min-h-screen">
            <h1 className="text-3xl font-bold mb-4">Image Processing & Object Detection</h1>
            <Card className="w-full max-w-2xl p-4 bg-gray-800 rounded-2xl shadow-lg">
                <CardContent className="flex flex-col items-center">
                    <label htmlFor="upload" className="cursor-pointer flex items-center gap-2 mb-4 text-blue-400">
                        <UploadCloud /> Upload Image
                    </label>
                    <Input id="upload" type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                    <Button className="mt-2 flex items-center gap-2 bg-green-500 hover:bg-green-600" onClick={startCamera}>
                        <Camera /> Use Camera
                    </Button>
                    {streaming && (
                        <>
                            <video ref={videoRef} autoPlay className="w-full max-w-lg mt-4 rounded-xl" />
                            <Button className="mt-2 bg-yellow-500 hover:bg-yellow-600" onClick={captureImage}>Capture Image</Button>
                            <Button className="mt-2 bg-red-500 hover:bg-red-600" onClick={stopCamera}>Stop Camera</Button>
                        </>
                    )}
                    <canvas ref={canvasRef} className="hidden" />
                    {image && <img src={image} alt="Captured" className="w-full rounded-xl mt-2" />}
                </CardContent>
            </Card>
            <Tabs defaultValue="original" className="w-full max-w-2xl mt-6">
                <TabsList className="flex justify-center bg-gray-700 p-2 rounded-xl">
                    {['original', 'grayscale', 'infrared', 'xray'].map((toneType) => (
                        <TabsTrigger key={toneType} value={toneType} onClick={() => setTone(toneType)}>
                            {toneType.charAt(0).toUpperCase() + toneType.slice(1)}
                        </TabsTrigger>
                    ))}
                </TabsList>
                <TabsContent value={tone} className="flex justify-center items-center mt-4">
                    {image ? (
                        <motion.img
                            key={tone}
                            src={image}
                            alt="Processed"
                            className="w-full max-w-lg rounded-xl"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                        />
                    ) : (
                        <p className="text-gray-400">No image selected</p>
                    )}
                </TabsContent>
            </Tabs>
            <div className="flex gap-4 mt-6">
                <Button className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600" onClick={() => alert("Processing image...")}>
                    <Scan /> Process Image
                </Button>
                <Button className="flex items-center gap-2 bg-purple-500 hover:bg-purple-600" onClick={downloadImage}>
                    <Download /> Download Image
                </Button>
            </div>
        </div>
    );
} 