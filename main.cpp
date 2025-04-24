#include "sgm.h"


int main(int argc, char **argv)
{

    if (argc != 8)
    {
        cerr << "Usage: " << argv[0] << " <right image> <left image> <monocular_right> <monocular_left>  <gt disparity map> <output image file> <disparity range> " << endl;
        return -1;
    }

    char *firstFileName = argv[1];
    char *secondFileName = argv[2];
    char *monoRightFileName = argv[3];
    char *monoLeftFileName = argv[4];
    char *gtFileName = argv[5];
    char *outputFileName = argv[6];
    unsigned int disparityRange = atoi(argv[7]);

    cv::Mat firstImage;
    cv::Mat secondImage;
    cv::Mat monoRight;
    cv::Mat monoLeft;
    cv::Mat gt;
    cv::Mat refined_disp_right;

    firstImage = cv::imread(firstFileName, IMREAD_GRAYSCALE);
    secondImage = cv::imread(secondFileName, IMREAD_GRAYSCALE);
    monoRight = cv::imread(monoRightFileName, IMREAD_GRAYSCALE);
    monoLeft = cv::imread(monoLeftFileName, IMREAD_GRAYSCALE);
    gt = cv::imread(gtFileName, IMREAD_GRAYSCALE);

    if (!firstImage.data || !secondImage.data)
    {
        cerr << "Could not open or find one of the images!" << endl;
        return -1;
    }

    cout << "******** round 1/2: SGM without left mono refinement ********" << endl;
    {
        sgm::SGM sgm(disparityRange);
        sgm.set(firstImage, secondImage, monoRight, monoLeft);
        sgm.compute_disparity();
        sgm.save_disparity(outputFileName);
        // sgm.save_confidence(outputFileName);
        std::cerr << "Right Image MSE error: " << sgm.compute_mse(gt) << std::endl;
    }

    cout << "******** round 2/2: SGM with left mono refinement ********" << endl;

    // using left mono to refine the right mono disparity map
    refineRightDisparity(monoLeft, monoRight, refined_disp_right, disparityRange, 120.0f);
    fillHolesInDisparity(refined_disp_right);
    imwrite("refined_disp_right.png", refined_disp_right);

    sgm::SGM sgm_refined(disparityRange);
    sgm_refined.set(firstImage, secondImage, refined_disp_right, monoLeft);
    sgm_refined.compute_disparity();

    
    // to extract base name without extension
    char refinedOutputName[256];
    std::string baseFileName = outputFileName;
    size_t lastDot = baseFileName.find_last_of(".");
    if (lastDot != std::string::npos)
    {
        baseFileName = baseFileName.substr(0, lastDot);
    }
    sprintf(refinedOutputName, "%s_refined%s",
            baseFileName.c_str(),
            (lastDot != std::string::npos) ? outputFileName + lastDot : "");

    sgm_refined.save_disparity(refinedOutputName);

    std::cerr << "Right Image MSE error with left mono refinement: " << sgm_refined.compute_mse(gt) << std::endl;

    return 0;
}
