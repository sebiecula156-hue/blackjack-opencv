#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// Funcție pentru a ordona punctele: [sus-stanga, sus-dreapta, jos-dreapta, jos-stanga]
void orderPoints(std::vector<cv::Point2f>& pts) {
    std::sort(pts.begin(), pts.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; });
    std::vector<cv::Point2f> top = { pts[0], pts[1] };
    std::vector<cv::Point2f> bottom = { pts[2], pts[3] };
    
    std::sort(top.begin(), top.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });
    std::sort(bottom.begin(), bottom.end(), [](cv::Point2f a, cv::Point2f b) { return a.x > b.x; });
    
    pts = { top[0], top[1], bottom[0], bottom[1] };
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    // Setări opționale pentru luminozitate (ajustează valorile dacă e prea întunecat/luminos)
    cap.set(cv::CAP_PROP_BRIGHTNESS, 100); 

    cv::Mat cornerThresh;
    cv::Mat frame, gray, blurred, edges;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Procesare imagine
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 0);
        cv::Canny(blurred, edges, 30, 100);
        cv::dilate(edges, edges, kernel); // Unim liniile întrerupte

        // Găsirea contururilor
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            
            // Filtru mărime: destul de mare pentru o carte, dar nu cât tot ecranul
            if (area > 5000 && area < (frame.cols * frame.rows * 0.7)) {
                double peri = cv::arcLength(contour, true);
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, 0.04 * peri, true);

                // Verificăm dacă are 4 colțuri și este o formă convexă (fără scobituri)
                if (approx.size() == 4 && cv::isContourConvex(approx)) {
                    
                    // Filtru Aspect Ratio (Lățime vs Înălțime)
                    cv::Rect r = cv::boundingRect(approx);
                    float ratio = (float)r.width / r.height;

                    // O carte are raport de ~0.7 (vertical) sau ~1.4 (orizontal)
                    if ((ratio > 0.4 && ratio < 0.9) || (ratio > 1.1 && ratio < 1.9)) {
                        
                        // Desenăm conturul pe imaginea principală
                        cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);

                        // --- WARP PERSPECTIVE (Îndreptarea cărții) ---
                        std::vector<cv::Point2f> src;
                        for(auto p : approx) src.push_back(cv::Point2f(p.x, p.y));
                        orderPoints(src);

                        // Dimensiuni finale pentru cartea scanată
                        std::vector<cv::Point2f> dst = { {0,0}, {200,0}, {200,300}, {0,300} };
                        cv::Mat M = cv::getPerspectiveTransform(src, dst);
                        cv::Mat warped;
                        cv::warpPerspective(frame, warped, M, cv::Size(200, 300));

                        // Decupăm colțul stânga-sus, luăm cam 20% din lățime și 35% din înălțime
                        cv::Rect roi(5, 5, 45, 80); 
                        cv::Mat cardCorner = warped(roi);
                        
                        // Îl facem alb-negru clar (Thresholding)
                        cv::Mat cornerGray;
                        cv::cvtColor(cardCorner, cornerGray, cv::COLOR_BGR2GRAY);
                        cv::adaptiveThreshold(cornerGray, cornerThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

                        cv::imshow("Corner Debug", cornerThresh);
                        cv::imshow("Cartea Scanata", warped);
                    }
                }
            }
        }

        cv::imshow("Blackjack Vision", frame);
        cv::imshow("Edges (Debug)", edges);

        int key = cv::waitKey(1);

        // Salvează imaginea
        if (key == 's'){
            std::string rankName;
            std::cout << "Introduceti rangul cartii: ";
            std::cin >> rankName;

            std::string fullPath = "D:/autoclicker/chestii/blackjack/templates/rank_" + rankName + ".jpg";
            cv::imwrite(fullPath, cornerThresh);

            std::cout << "Imaginea a fost salvată ca '" << fullPath << "'." << std::endl;
        }
        else if(key >= 0) break; // Se închide programul la orice apăsare de tastă
    }

    return 0;
}