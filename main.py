import time as t
import cv2

from processors import image_process, solve_sudoku
from models import wrap_model

def set_camera():
    frameWidth = 960
    frameHeight = 720

    cap = cv2.VideoCapture(0)
    frame_rate = 30

    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    cap.set(10, 150)
    return frame_rate, cap

def draw_digits(result,warped, sudoku, boxes_quessed,corners,time):
    image_process.draw_digits_on_warped(warped, sudoku, boxes_quessed)
    img_result = image_process.unwarp_image(warped, result, corners, time)
    return img_result
                        
def main():
    frame_rate,cap = set_camera()
    my_model = wrap_model.build_model(None, False, None, "model.hdf5")

    prev = 0
    view = dict()
    while True:
        time_elapsed = t.time() - prev
        _, img = cap.read()

        if time_elapsed > 1. / frame_rate:
            prev = t.time()

            img_result = img.copy()
            img_corners = img.copy()

            processed_img = image_process.preprocess(img)
            corners = image_process.find_contours(processed_img, img_corners)

            if corners:
                warped, _ = image_process.warp_image(corners, img)
                warped_processed = image_process.preprocess(warped)

                vertical_lines, horizontal_lines = image_process.get_grid_lines(warped_processed)
                mask = image_process.create_grid_mask(vertical_lines, horizontal_lines)
                numbers = cv2.bitwise_and(warped_processed, mask)

                squares = image_process.split_into_squares(numbers)
                squares_processed = image_process.clean_squares(squares)

                boxes_quessed = image_process.recognize_digits(squares_processed, my_model)

                if boxes_quessed in view and view[boxes_quessed] is False:
                    continue
                
                if boxes_quessed in view:
                    img_result = draw_digits(img_result,warped, view[boxes_quessed][0], squares_processed, corners, view[boxes_quessed][1])
                else:
                    solved_puzzle, time = solve_sudoku.solve_wrapper(boxes_quessed)
                    
                    if solved_puzzle is not None:
                        img_result = draw_digits(img_result, warped, solved_puzzle, squares_processed, corners, time)
                        view[boxes_quessed] = [solved_puzzle, time]
                    else:
                        view[boxes_quessed] = False

        cv2.imshow('window', img_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
