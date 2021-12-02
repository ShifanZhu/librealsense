// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include "darknet.h"

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering

#include "../wrappers/opencv/cv-helpers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

void imgConvert(const cv::Mat &img, float *dst)
{
    uchar *data = img.data;
    int h=img.rows;
    int w=img.cols;
    int c=img.channels();

    for(int k =0;k<c;++k)
    {
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
            {
                dst[k*w*h+i*w+j] = data[(i*w+j)*c+k]/255.0;
            }
    }
}
void resizeInner(float *src, float *dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    float* part;
    size_t sizePa = dstWidth*srcHeight*3*sizeof(float);
    part = (float*)malloc(sizePa);

    float w_scale = (float)(srcWidth -1)/(dstWidth -1);
    float h_scale = (float)(srcHeight - 1)/(dstHeight -1);

    for(int k =0;k<3;k++)
        for(int r=0;r<srcHeight;r++)
            for(int c=0;c<dstWidth;c++)
            {
                float val=0;
                if(c==dstWidth-1 || srcWidth ==1)
                {
                    val =src[k*srcWidth*srcHeight+r*srcWidth+srcWidth-1];
                }
                else
                {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx=sx-ix;
                    val =(1-dx)*src[k*srcWidth*srcHeight+r*srcWidth+ix] + dx*src[k*srcWidth*srcHeight+r*srcWidth+ix+1];
                }
                part[k*srcHeight*dstWidth + r*dstWidth +c] =val;
            }
    for(int k=0;k<3;k++)
        for(int r =0;r<dstHeight;++r)
        {
            float sy = r*h_scale;
            int iy =(int) sy;
            float dy = sy-iy;
            for(int c =0;c<dstWidth;++c)
            {
                float val =(1-dy)*part[k*dstWidth*srcHeight+iy*dstWidth+c];
                dst[k*dstWidth*dstHeight + r*dstWidth+c]=val;
            }
            if(r==dstHeight-1 || srcHeight ==1)
                continue;
            for(int c=0;c<dstWidth;++c)
            {
                float val =dy* part[k*dstWidth*srcHeight+(iy+1)*dstWidth+c];
                dst[k*dstWidth*dstHeight+r*dstWidth+c]+=val;
            }
        }
    free(part);

}

void imgResize(float *src, float *dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    int new_w = srcWidth;
    int new_h = srcHeight;
    if(((float)dstWidth/srcWidth)<((float)dstHeight/srcHeight))
    {
        new_w = dstWidth;
        new_h = (srcHeight * dstWidth)/srcWidth;
    }
    else {
        new_h = dstHeight;
        new_w = (srcWidth*dstHeight)/srcHeight;
    }

    float* ImgReInner;
    size_t sizeInner = new_w*new_h*3*sizeof(float);
    ImgReInner = (float*)malloc(sizeInner);
    resizeInner(src,ImgReInner,srcWidth,srcHeight,new_w,new_h);

    for(int i =0;i<dstWidth*dstHeight*3;i++)
        dst[i]=0.5;

    for(int k=0;k<3;k++)
        for(int y=0;y<new_h;y++)
            for(int x=0;x<new_w;x++)
            {
                float val = ImgReInner[k*new_w*new_h+y*new_w+x];
                dst[k*dstHeight*dstWidth + ((dstHeight-new_h)/2+y)*dstWidth +(dstWidth-new_w)/2+x]=val;
            }
    free(ImgReInner);
}

float colors[6][3] = {{1,0,1},{0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0}};

float get_color(int c, int x, int max)          //set the color of lines
{
    float ratio = ((float)x/max)*5;
    int i=floor(ratio);
    int j=ceil(ratio);
    ratio -= i;
    float r=(1-ratio)*colors[i][c]+ratio*colors[j][c];
    return r;
}

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char * argv[]) try
{
    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Capture Example");

    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;
    // Declare rates printer for showing streaming rates of the enabled streams.
    rs2::rates_printer printer;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    std::string cfgfile = "/home/zh/code/darknet/cfg/yolov3-obj-door.cfg";                    //cfg_file
    std::string weightfile = "/home/zh/data/yolo-weight/yolo-obj-door.weights";    //weight_file


    network *net;
    net = load_network((char*) cfgfile.c_str(),(char*) weightfile.c_str(),0);  //build the net
    set_batch_network(net,1);
    // ifstream classNamesFile("/home/zh/code/darknet/data/voc.names");                 //classname_file
    std::ifstream classNamesFile("/home/zh/code/darknet/data/obj-door.names");                 //classname_file
    std::vector<std::string> classNamesVec;
    
    if(classNamesFile.is_open())
    {
        std::string className = "";
        while(getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }


    float thresh = 0.1;                                                         //if >thresh,then detect
    float nms = 0.35;                                                           //nms thresh
    int classes = 4; 
    std::vector<cv::Rect> boxes;
    std::vector<int> classNames;

    const auto window_name_depth = "Display depth Image";
    const auto window_name_color = "Display color Image";
    const auto window_name_door = "Display detected door";
    namedWindow(window_name_depth, cv::WINDOW_AUTOSIZE);
    namedWindow(window_name_color, cv::WINDOW_AUTOSIZE);
    namedWindow(window_name_door, cv::WINDOW_AUTOSIZE);

    rs2::align align_to(RS2_STREAM_COLOR);

    // Start streaming with default recommended configuration
    // The default video configuration contains Depth and Color streams
    // If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
    pipe.start();

    while (app) // Application still alive?
    {
        rs2::frameset data = pipe.wait_for_frames().    // Wait for next set of frames from the camera
                             apply_filter(printer).     // Print each enabled stream frame rate
                             apply_filter(color_map);   // Find and colorize the depth data
        
        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);

        // The show method, when applied on frameset, break it to frames and upload each frame into a gl textures
        // Each texture is displayed on different viewport according to it's stream unique id
        app.show(data);

                // Query frame size (width and height)
        const int w = depth.as<rs2::video_frame>().get_width();
        const int h = depth.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        cv::Mat image_depth(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
        // cv::Mat image_color(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);


        rs2::frameset aligned_set = align_to.process(data);
        // rs2::frame depth = aligned_set.get_depth_frame();
        auto image_color = frame_to_mat(aligned_set.get_color_frame());

        cv::Mat current_frame = image_color;
        cv::Mat rgbImg = current_frame;
        // cv::cvtColor(current_frame, rgbImg, cv::COLOR_BGR2RGB); // BGR2RGB

        float* srcImg;
        size_t srcSize = rgbImg.rows*rgbImg.cols*3*sizeof(float);
        srcImg=(float*)malloc(srcSize);
        imgConvert(rgbImg,srcImg); //将图像转为yolo形式

        float* resizeImg;
        size_t resizeSize = net->w*net->h*3*sizeof(float);
        resizeImg=(float*)malloc(resizeSize);
        imgResize(srcImg,resizeImg,rgbImg.cols,rgbImg.rows,net->w,net->h); //缩放图像

        network_predict(net, resizeImg); //network_predict costly!

        int nboxes=0;
        thresh = 0.5;
        detection *dets=get_network_boxes(net,rgbImg.cols,rgbImg.rows,thresh,0.5,0,1,&nboxes);

        if(nms)                                                                 //nms
        {
            do_nms_sort(dets,nboxes,classes,nms);
        }

        boxes.clear();
        classNames.clear();
        for(int i=0;i<nboxes;i++)
        {
            bool flag=0;
            int className;
            for(int j=0;j<classes;j++)
            {
                // if(j==0 || j==1 || j==2 || j==56)
                // if(j==2)
                {//only person, bicycle, car and chair left
                    if(dets[i].prob[j]>thresh)
                    {
                        thresh = dets[i].prob[j];
                        // if(!flag)
                        // {
                            flag=1;
                            className=j;
                        // }
                    }
                }
            }
            if(flag)
            {
                int left = (dets[i].bbox.x - dets[i].bbox.w/2.)*rgbImg.cols;
                int right = (dets[i].bbox.x + dets[i].bbox.w/2.)*rgbImg.cols;
                int top = (dets[i].bbox.y - dets[i].bbox.h/2.)*rgbImg.rows;
                int bot = (dets[i].bbox.y + dets[i].bbox.h/2.)*rgbImg.rows;

                if(left < 0)                left = 0;
                if(right > rgbImg.cols -1)   right = rgbImg.cols - 1;
                if(top < 0)                 top = 0;
                if(bot > rgbImg.rows-1)      bot = rgbImg.rows-1;

                if(fabs(bot-top) < 50 || fabs(left-right) < 50)
                    continue;

                cv::Rect box(left, top, fabs(left-right),fabs(bot-top));
                boxes.push_back(box);                                               //save boxes
                classNames.push_back(className);
            }
        }

        free_detections(dets,nboxes);

        std::vector<cv::Point3d> points_box;
        for(unsigned int i=0;i<boxes.size();i++)
        {
            int offset = classNames[i]*123457 % 80;
            float red   = 255*get_color(2,offset,80);
            float green = 255*get_color(1,offset,80);
            float blue  = 255*get_color(0,offset,80);

            cv::rectangle(rgbImg,cv::Point(boxes[i].x,boxes[i].y),
                        cv::Point((boxes[i].x+boxes[i].width),(boxes[i].y+boxes[i].height)),
                        cv::Scalar(blue,green,red),2,8,0);                                //draw boxes

            
                cv::String label = cv::String(classNamesVec[classNames[i]]);
                int baseLine = 0;
                cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5,1, &baseLine);
                putText(rgbImg,label, cv::Point(boxes[i].x,boxes[i].y+labelSize.height),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(red,blue,green),2);
                cv::Point3d temp_box(boxes[i].x+boxes[i].width/2.0, boxes[i].y+boxes[i].height/2.0, 0);
                // Point3d temp_box(boxes[i].x+boxes[i].width/2.0, boxes[i].y+boxes[i].height/2.0, color_type);
                points_box.push_back(temp_box);
        }





        // Update the window with new data
        imshow(window_name_depth, image_depth);
        imshow(window_name_color, rgbImg);
        imshow(window_name_door, rgbImg);
        cv::waitKey(1);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}