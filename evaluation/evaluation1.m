clear
clc
close all


imageFolder = 'gt/vender'; % Generated image folder
gtFolder = 'ours/vender'; % Ground truth image folder

% Read image files from the specified folder
imageFiles = dir(fullfile(imageFolder, '*.png')); % Change file extension if needed
gtFiles = dir(fullfile(gtFolder, '*.png'));

% Initialize variables to store PSNR and SSIM values
psnrValues = zeros(length(imageFiles), 1);
ssimValues = zeros(length(imageFiles), 1);

% Loop through each image file
for i = 1:length(imageFiles)
    % Read the current image
    imagePath = fullfile(imageFolder, imageFiles(i).name);
    image = imread(imagePath);
    
    % Read the corresponding ground truth image
    gtImagePath = fullfile(gtFolder, gtFiles(i).name);
    gtImage = imread(gtImagePath);
    
    imageGray = rgb2gray(image);
    gtGray = rgb2gray(gtImage);
    
    % Calculate SSIM
    ssimValues(i) = ssim(image, gtImage); % SSIM calculation using built-in function
    
    % Calculate PSNR
    psnrValues(i) = psnr(imageGray, gtGray);
end

% Display the PSNR and SSIM values for each image
for i = 1:length(imageFiles)
    fprintf('Image: %s\n', imageFiles(i).name);
    fprintf('PSNR: %.2f dB\n', psnrValues(i));
    fprintf('SSIM: %.4f\n\n', ssimValues(i));
end

SSIM_clear = mean(ssimValues);
PSNR_clear = mean(psnrValues);