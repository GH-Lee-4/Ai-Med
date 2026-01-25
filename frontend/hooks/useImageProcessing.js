export function useImageProcessing(image, setImage) {
    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = () => setImage(reader.result);
            reader.readAsDataURL(file);
        }
    };

    const downloadImage = () => {
        if (image) {
            const link = document.createElement("a");
            link.href = image;
            link.download = "processed_image.png";
            link.click();
        }
    };

    return {
        handleImageUpload,
        downloadImage
    };
} 