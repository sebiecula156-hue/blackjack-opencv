#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// Funcție pentru a ordona punctele: [sus-stanga, sus-dreapta, jos-dreapta, jos-stanga]
void orderPoints(std::vector<cv::Point2f>& pts) {
    std::sort(pts.begin(), pts.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; });
    std::vector<cv::Point2f> top = {pts[0], pts[1]};
    std::vector<cv::Point2f> bottom = {pts[2], pts[3]};
    std::sort(top.begin(), top.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });
    std::sort(bottom.begin(), bottom.end(), [](cv::Point2f a, cv::Point2f b) { return a.x > b.x; });
    pts = {top[0], top[1], bottom[0], bottom[1]};
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame, gray, thresh, kernel;
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

        // REPARARE: Îngroșăm liniile albe ca să unim conturul unde e degetul
        cv::dilate(thresh, thresh, kernel, cv::Point(-1, -1), 1);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 5000 && area < 150000) {
                double peri = cv::arcLength(contour, true);
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, 0.02 * peri, true);

                if (approx.size() == 4) {
                    // Desenăm conturul pe imaginea principală
                    cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 3);

                    // WARP PERSPECTIVE: Îndreptăm imaginea cărții
                    std::vector<cv::Point2f> pts_src, pts_dst;
                    for(auto p : approx) pts_src.push_back(cv::Point2f(p.x, p.y));
                    orderPoints(pts_src);

                    // Dimensiuni standard pentru o carte scanată (ex: 200x300 pixeli)
                    pts_dst = { {0,0}, {200,0}, {200,300}, {0,300} };

                    cv::Mat M = cv::getPerspectiveTransform(pts_src, pts_dst);
                    cv::Mat warped;
                    cv::warpPerspective(frame, warped, M, cv::Size(200, 300));

                    cv::imshow("Cartea Scanata", warped);
                }
            }
        }
        cv::imshow("Blackjack Vision", frame);
        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}