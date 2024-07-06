#define NOMINMAX

#include <iostream>
#include <fstream>
#include <exception>
#include <map>
#include <vector>
#include <string>
#include <locale>
#include <codecvt>
#include <random>

#include <Windows.h>

#include <facerec/import.h>
#include <facerec/libfacerec.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "ConsoleArgumentsParser.h"

static const std::map<int, std::string> CvTypeToStr{ {CV_8U,"uint8_t"}, {CV_8S,"int8_t"},{CV_16U,"uint16_t"}, {CV_16S,"int16_t"},{CV_32S,"int32_t"}, {CV_32F,"float"}, {CV_64F,"double"} };

static std::vector<std::string> FileExt = { ".png",".bmp",".tif",".tiff",".jpg",".jpeg",".ppm",
											".PNG",".BMP",".TIF",".TIFFF",".JPG",".JPEG",".PPM" };

void convertMat2BSM(pbio::Context& bsmCtx, const cv::Mat& img, bool copy = false) {
	const cv::Mat& input_img = img.isContinuous() ? img : img.clone(); // setDataPtr requires continuous data
	
	size_t copy_sz = (copy || !img.isContinuous()) ? input_img.total() * input_img.elemSize() : 0;
	
	bsmCtx["format"] = "NDARRAY";
	bsmCtx["blob"].setDataPtr(input_img.data, copy_sz);
	bsmCtx["dtype"] = CvTypeToStr.at(input_img.depth());
	
	for (int i = 0; i < input_img.dims; ++i)
		bsmCtx["shape"].push_back(input_img.size[i]);
	
	bsmCtx["shape"].push_back(input_img.channels());
}

void toCSV(std::ofstream& csv, const pbio::Context& ioData) {
	auto obj = ioData["objects"];
	auto quality = obj[0]["quality"];
	
	csv << obj[0]["confidence"].getDouble()
		<<","<< static_cast<int>(quality["total_score"].getDouble() * 100)
		<<","<< (quality["is_sharp"].getBool() ? 1 : 0)
		<<","<< static_cast<int>(quality["sharpness_score"].getDouble() * 100)
		<<","<< (quality["is_evenly_illuminated"].getBool() ? 1 : 0)
		<<","<< (quality["no_flare"].getBool() ? 1 : 0)
		<<","<< (quality["is_left_eye_opened"].getBool() ? 1 : 0)
		<<","<< (quality["is_right_eye_opened"].getBool() ? 1 : 0)
		<<","<< (quality["is_rotation_acceptable"].getBool() ? 1 : 0)
		<<","<< (quality["not_masked"].getBool() ? 1 : 0)
		<<","<< (quality["is_neutral_emotion"].getBool() ? 1 : 0)
		<<","<< (quality["is_eyes_distance_acceptable"].getBool() ? 1 : 0)
		<<","<< (quality["eyes_distance"].getLong())
		<<","<< (quality["is_margins_acceptable"].getBool() ? 1 : 0)
		<<","<< (quality["is_not_noisy"].getBool() ? 1 : 0)
		<<","<< (quality["has_watermark"].getBool() ? 1 : 0)
		<<","<< static_cast<int>(quality["dynamic_range_score"].getDouble() * 100)
		<<","<< (quality["is_dynamic_range_acceptable"].getBool() ? 1 : 0)
		<< std::endl;
}

bool checkFileExt(const std::string filename) {
	for (const auto ext : FileExt) {
		if (std::strstr(filename.c_str(), ext.c_str()) != nullptr)
			return true;
	}

	return false;
}

std::vector<std::string> scanDir(const std::string& path) {
	std::vector<std::string> filenames;

	using convertType = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convertType, wchar_t> converter;

	const auto unicodePath = converter.from_bytes(path+"/*.*");

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = FindFirstFile(unicodePath.c_str(), &FindFileData);


	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (FindFileData.dwFileAttributes != FILE_ATTRIBUTE_DIRECTORY) {
				std::wstring filenameW(FindFileData.cFileName);
				const auto filename = converter.to_bytes(filenameW);

				if (checkFileExt(filename))
					filenames.push_back(filename);
			}
		} while (FindNextFile(hFind, &FindFileData));

		FindClose(hFind);
	}

	return filenames;
}

int main(int argc, char** argv) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	ConsoleArgumentsParser argParser(argc, argv);

	const std::string SDK_PATH = argParser.get<std::string>("--sdk-path", "C:/3DiVi_FaceSDK/3_22_0/");
	const std::string DLL_PATH = SDK_PATH + "bin/facerec.dll";
	const std::string CONFIG_PATH = SDK_PATH + "conf/facerec";
	const std::string LICENCE_PATH = SDK_PATH + "license/";

	const std::string imgDirPath = argParser.get<std::string>("--dir");
	int numProcessed = argParser.get<int>("--num-processed", 0);

	if (imgDirPath.empty()) {
		std::cout << "--dir is required" << std::endl;
		return -1;
	}

	auto images = scanDir(imgDirPath);

	if (images.size() <= numProcessed || numProcessed <= 0) {
		numProcessed = images.size();
	} else {
		//setting up RNG
		std::random_device dev;
		std::mt19937 rng(dev());
		using rngType = std::mt19937::result_type;

		//remove random files from vector
		for (size_t i = 0; images.size() > numProcessed; i++) {
			std::uniform_int_distribution<rngType> dist(0, images.size()-1);

			std::swap(images[dist(rng)], images.back());
			images.pop_back();
		}
	}

	std::ofstream result("result.csv", std::ios::out);
	result << "Confidence,"
		<<"totalScore,"
		<<"isSharp,"
		<<"sharpnessScore,"
		<<"isEvenlyIlluminated,"
		<<"illuminationScore,"
		<<"noFlare,"
		<<"isLeftEyeOpened,"
		<<"isRightEyeOpened,"
		<<"isRotationAcceptable,"
		<<"notMasked,"
		<<"isNeutralEmotion,"
		<<"isEyesDistanceAcceptable,"
		<<"eyesDistance,"
		<<"isMarginsAcceptable,"
		<<"isNotNoisy,"
		<<"hasWatermark,"
		<<"dynamicRangeScore,"
		<<"isDynamicRangeAcceptable"
		<< std::endl;

	try {
		pbio::FacerecService::Ptr service;
		service = pbio::FacerecService::createService(DLL_PATH, CONFIG_PATH, LICENCE_PATH);
		
		auto configCtx = service->createContext();

		//Detector -> Fitter -> Quality
		configCtx["unit_type"] = "FACE_DETECTOR";
		auto detectorBlock = service->createProcessingBlock(configCtx);

		configCtx["unit_type"] = "FACE_FITTER";
		auto fitterBlock = service->createProcessingBlock(configCtx);

		configCtx["unit_type"] = "QUALITY_ASSESSMENT_ESTIMATOR";
		configCtx["config_name"] = "quality_assessment.xml";
		auto qualityBlock = service->createProcessingBlock(configCtx);

		size_t i = 0;
		for (const auto& filename : images) {
			auto path = imgDirPath + "/" + filename;
			std::cout << "Processing: " << path << "\t(" << ++i << "/" << images.size() <<")" << std::endl;
			
			cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
			cv::Mat input_image;
			cv::cvtColor(img, input_image, cv::COLOR_BGR2RGB);

			auto ioData = service->createContext();
			auto imgCtx = service->createContext();
			convertMat2BSM(imgCtx, input_image);
			ioData["image"] = imgCtx;

			detectorBlock(ioData);

			if (ioData["objects"].size() > 0) {
				fitterBlock(ioData);
				qualityBlock(ioData);
				
				toCSV(result, ioData);
			}
		}
	} catch (const pbio::Error& e){
		std::cerr << "Facerec exception: " << e.what() << "\nCode: " << std::hex << e.code() << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "C++ exception: " << e.what() << std::endl;
	}

	return 0;
}