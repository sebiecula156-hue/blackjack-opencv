#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

// Calculeaza IOU
float computeIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    if (intersection.area() == 0) return 0.f;
    float unionArea = (float)(a.area() + b.area() - intersection.area());
    return (float)intersection.area() / unionArea;
}

// Functie pentru a ordona punctele: [sus-stanga, sus-dreapta, jos-dreapta, jos-stanga]
void orderPoints(std::vector<cv::Point2f>& pts) {
    std::sort(pts.begin(), pts.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; });

    std::vector<cv::Point2f> top = { pts[0], pts[1] };
    std::vector<cv::Point2f> bottom = { pts[2], pts[3] };

    std::sort(top.begin(), top.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });
    std::sort(bottom.begin(), bottom.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });

    pts = { top[0], top[1], bottom[0], bottom[1] };
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    // Ajustari camera
    cap.set(cv::CAP_PROP_BRIGHTNESS, 150);

    cv::Mat frame, cornerThresh;

    // Variabile pentru stabilizare intre frame-uri
     cv::Rect prevBoundingBox;
     double accumulatedTime = 0.0;
     auto lastSeenTime = std::chrono::steady_clock::now();
     bool isStable = false;
     bool timerRunning = false;
     const float IOU_THRESHOLD = 0.60f;
     const double STABLE_SECONDS = 3.0;
     const double PAUSE_TOLERANCE  = 0.5;
     const int WARP_W = 200, WARP_H = 300;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        bool cardDetected = false;

        // HSV mask
        cv::Mat hsv, whiteMask;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // Filtru pentru alb
        cv::inRange(hsv, 
            cv::Scalar(0,   0, 160),   // H_min, S_min, V_min 
            cv::Scalar(180, 60, 255),  // H_max, S_max, V_max
            whiteMask);
        
        // Morfologie pentru a curata masca
        cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_OPEN,  morphKernel); // elimina zgomot
        cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_CLOSE, morphKernel); // umple gauri

        // Gasirea contururilor pe masca alba
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(whiteMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Debug
        cv::imshow("Edges (Debug)", whiteMask);

        // Variabile pentru a retine cel mai bun contur gasit
        cv::RotatedRect bestRect;;
        double bestArea = 0.0;
        bool foundCard = false;

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 8000) continue;

            cv::RotatedRect rRect = cv::minAreaRect(contour);
            float w = rRect.size.width;
            float h = rRect.size.height;
            if (w < 1 || h < 1) continue;
            
            // Normalizam astfel incat ratio > 1 intotdeauna
            float ratio = (w > h) ? w / h : h / w;

            if (ratio < 1.1f || ratio > 2.2f) continue;

            // Retinem cel mai mare candidat valid
            if (area > bestArea) {
                bestArea = area;
                bestRect = rRect;
                foundCard = true;
            }
        }

        // Procesam doar daca am gasit un candidat valid
        if (foundCard) {
            // Extragem cele 4 colturi ale dreptunghiului rotit
            cv::Point2f corners[4];
            bestRect.points(corners);

            // Construim bestCountour pentru drawContours
            std::vector<cv::Point> bestContour;
            for (auto& c : corners) bestContour.push_back(cv::Point((int)c.x, (int)c.y));

            cv::Rect boundingBox = cv::boundingRect(bestContour);

            auto now = std::chrono::steady_clock::now();
            float iou = prevBoundingBox.area() > 0 ? computeIoU(boundingBox, prevBoundingBox) : 0.f;

            if (iou >= IOU_THRESHOLD) {
                // Cartea e in aceeasi pozitie — acumulam timp
                if (timerRunning) {
                    double delta = std::chrono::duration<double>(now - lastSeenTime).count();
                    accumulatedTime += delta;
                }
                timerRunning = true;
                isStable     = (accumulatedTime >= STABLE_SECONDS);
            } else {
                // Carte noua / mutata semnificativ — reset complet
                accumulatedTime = 0.0;
                timerRunning = false;
                isStable = false;
            }
            prevBoundingBox = boundingBox;
            lastSeenTime = now;

            // Afisam progresul countdown-ului pe ecran
            double remaining = std::max(0.0, STABLE_SECONDS - accumulatedTime);
            std::string countdownText = isStable
                ? "GATA!"
                : "Stabilitate: " + std::to_string((int)std::ceil(remaining)) + "s";
            cv::Scalar textColor = isStable ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
            cv::putText(frame, countdownText,
                cv::Point(boundingBox.x, boundingBox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);

            // Desenam conturul detectat pe imaginea principala
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{bestContour}, -1, textColor, 2);

            // Warp perspective
            std::vector<cv::Point2f> src(corners, corners + 4);
            orderPoints(src);

            std::vector<cv::Point2f> dst = {
                {0.f,           0.f          },
                {(float)WARP_W, 0.f          },
                {0.f,           (float)WARP_H},
                {(float)WARP_W, (float)WARP_H}
            };
            cv::Mat M = cv::getPerspectiveTransform(src, dst);
            cv::Mat warped;
            cv::warpPerspective(frame, warped, M, cv::Size(WARP_W, WARP_H));

            // Decupare colt pentru citire (proportional din dimensiunile warp-ului)
            int padX = (int)(WARP_W * 0.025);
            int padY = (int)(WARP_H * 0.017);
            int roiW = (int)(WARP_W * 0.225);
            int roiH = (int)(WARP_H * 0.283);
            cv::Rect roi(padX, padY, roiW, roiH);

            if (warped.cols >= padX + roiW && warped.rows >= padY + roiH) {
                cv::Mat cardCorner = warped(roi);
                cv::Mat cornerGray;
                cv::cvtColor(cardCorner, cornerGray, cv::COLOR_BGR2GRAY);
                cv::adaptiveThreshold(cornerGray, cornerThresh, 255,
                    cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

                cardDetected = true;

                cv::imshow("Corner Debug", cornerThresh);
            }

            cv::imshow("Cartea Scanata", warped);
        }

        // Daca nu s-a detectat nicio carte, pause timer in loc de reset
        if (!cardDetected) {
            auto now = std::chrono::steady_clock::now();
            double missingFor = std::chrono::duration<double>(now - lastSeenTime).count();
            if (missingFor > PAUSE_TOLERANCE) {
                // Cartea a disparut prea mult — reset complet
                prevBoundingBox = cv::Rect();
                accumulatedTime = 0.0;
                timerRunning    = false;
                isStable        = false;
            }
        }

        cv::imshow("Blackjack Vision", frame);
        
        // Salvare imagine
        int key = cv::waitKey(1);
        if (key == 's' || key == 'S') {
            if (cardDetected && isStable && !cornerThresh.empty()) {
                std::string rankName;
                std::cout << "Introduceti numele/rangul cartii: ";
                std::cin >> rankName;

                std::string fullPath = "D:/autoclicker/chestii/blackjack/templates/rank_" + rankName + ".jpg";
                cv::imwrite(fullPath, cornerThresh);
                std::cout << "Salvat: " << fullPath << std::endl;
            } else {
                std::cout << "Nicio carte detectata in frame-ul curent. Nu s-a salvat nimic." << std::endl;
            }
        } else if (key == 27) {
            break;
        }
    }
    return 0;
}