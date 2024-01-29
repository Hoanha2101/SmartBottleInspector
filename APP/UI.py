from library import *
from AI import CHECK_BOTTLE_AI, CHECK_WATER_LEVEL_AI, CHECK_LABEL_AI
from utils import *
import tensorrt as trt
# import modules.utils as utils
# from modules.autobackend import AutoBackend

CLEAN_CSV_BOTTLE()
CLEAN_CSV_WATER_LEVEL()
CLEAN_CSV_LEVEL()

pygame.init()

screen_width = 1536
screen_height = 800
# Kích thước cửa sổ hiển thị camera
camera_height = 500
camera_width = 350

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("DETECT ERRORS")

font = pygame.font.Font(None, 36)
border_radius_button = 30

# Thiết lập camera 1
camera_1 = cv2.VideoCapture(1)

# Thiết lập camera 2
camera_2 = cv2.VideoCapture(3)

# Thiết lập camera 3
camera_3 = cv2.VideoCapture(2)

#Logo ))
logo_fpt_path = os.path.join("APP/image_set/logofptuniversity.png")
logo_fpt_surface = pygame.image.load(logo_fpt_path)
logo_fpt_surface = pygame.transform.scale(logo_fpt_surface, (150, 58))

#Exit button ))
exit_path = os.path.join("APP/image_set/exit.png")
exit_surface = pygame.image.load(exit_path)
exit_surface = pygame.transform.scale(exit_surface, (40, 40))
exit_clickable_area = pygame.Rect(20, screen_height - 70, 40, 40)


#setting button ))
setting_path = os.path.join("APP/image_set/settings.png")
setting_surface = pygame.image.load(setting_path)
setting_surface = pygame.transform.scale(setting_surface, (40, 40))
setting_clickable_area = pygame.Rect(20, screen_height - 120, 40, 40)


# Button Start - End
button_start_rect = pygame.Rect(1380, 700, 120, 50)  
button_start_color = (0,128,0)
button_start_text = font.render("START", True, (255, 255, 255))
text_start_rect = button_start_text.get_rect(center=button_start_rect.center)

# Các thành phần trong hộp thông tin
# hộp để show thông tin cho camera 1
square_rect_1 = pygame.Rect(30, 400, 460, 150) 
id_title_1 = font.render("ID: ", True, (0, 0, 0))
id_title_1_rect = id_title_1.get_rect(center=(70, 450))
status_title_1 = font.render("STATUS: ", True, (0, 0, 0))
status_title_1_rect = status_title_1.get_rect(center=(103, 500))
#Thông tin show - action
id_info_color_1 = (0,200,0)
id_info_error_text_1 = font.render("-", True, id_info_color_1)
id_info_error_rect_1 = id_info_error_text_1.get_rect(center=(300, 450))
info_color_1 = (0,200,0)
info_error_text_1 = font.render("-", True, info_color_1)
info_error_rect_1 = info_error_text_1.get_rect(center=(300, 500))

#Thẻ cha chứa On button và OFF button camera 1
div_ON_OFF_camera_1 = pygame.draw.rect(screen,(255, 0, 255),(145,580,200,40))
## ------------On button set camera 1 on
switch_on_text_camera_1 = font.render("ON", True, (0,0,0))
switch_on_rect_camera_1 = switch_on_text_camera_1.get_rect(center=(191,600))
switch_on_rect_box_camera_1 = pygame.draw.rect(screen,(255, 255, 255),(145,580,100,40))

## -------------OFF button set camera 1 off
switch_off_text_camera_1 = font.render("OFF", True, (0,0,0))
switch_off_rect_camera_1 = switch_off_text_camera_1.get_rect(center=(295,600))
switch_off_rect_box_camera_1 = pygame.draw.rect(screen,(255, 255, 255),(245,580,100,40))

# Đèn thông báo status
status_light_rect_camera_1 = pygame.Rect(70, 590, 20, 20) 
status_light_color_camera_1 = (255,0,0)

'''
    =================================================================================================
'''

# hộp để show thông tin cho camera 2
square_rect_2 = pygame.Rect(538, 400, 460, 150)
id_title_2 = font.render("ID: ", True, (0, 0, 0))
id_title_2_rect = id_title_2.get_rect(center=(578, 450))
status_title_2 = font.render("STATUS: ", True, (0, 0, 0))
status_title_2_rect = status_title_2.get_rect(center=(611, 500))
#Thông tin show - action
id_info_color_2 = (0,200,0)
id_info_error_text_2 = font.render("-", True, id_info_color_2)
id_info_error_rect_2 = id_info_error_text_2.get_rect(center=(808, 450))
info_color_2 = (0,200,0)
info_error_text_2 = font.render("-", True, info_color_2)
info_error_rect_2 = info_error_text_2.get_rect(center=(808, 500))

#Thẻ cha chứa On button và OFF button camera 2
div_ON_OFF_camera_2 = pygame.draw.rect(screen,(255, 0, 255),(668,580,200,40))
## ------------On button set camera 2 on
switch_on_text_camera_2 = font.render("ON", True, (0,0,0))
switch_on_rect_camera_2 = switch_on_text_camera_2.get_rect(center=(714,600))
switch_on_rect_box_camera_2 = pygame.draw.rect(screen,(255, 255, 255),(668,580,100,40))

## -------------OFF button set camera 2 off
switch_off_text_camera_2 = font.render("OFF", True, (0,0,0))
switch_off_rect_camera_2 = switch_off_text_camera_2.get_rect(center=(818,600))
switch_off_rect_box_camera_2 = pygame.draw.rect(screen,(255, 255, 255),(768,580,100,40))

# Đèn thông báo status
status_light_rect_camera_2 = pygame.Rect(578, 590, 20, 20) 
status_light_color_camera_2 = (255,0,0)

'''
    =================================================================================================
'''

# hộp để show thông tin cho camera 3
square_rect_3 = pygame.Rect(screen_width - 490, 400, 460, 150) 
id_title_3 = font.render("ID: ", True, (0, 0, 0))
id_title_3_rect = id_title_3.get_rect(center=(screen_width - 450, 450))
status_title_3 = font.render("STATUS: ", True, (0, 0, 0))
status_title_3_rect = status_title_3.get_rect(center=(screen_width - 417, 500))
#Thông tin show - action
id_info_color_3 = (0,200,0)
id_info_error_text_3 = font.render("-", True, id_info_color_3)
id_info_error_rect_3 = id_info_error_text_3.get_rect(center=(screen_width - 220, 450))
info_color_3 = (0,200,0)
info_error_text_3 = font.render("-", True, info_color_3)
info_error_rect_3 = info_error_text_3.get_rect(center=(screen_width - 220, 500))


#Thẻ cha chứa On button và OFF button camera 3
div_ON_OFF_camera_3 = pygame.draw.rect(screen,(255, 0, 255),(1176,580,200,40))
## ------------On button set camera 3 on
switch_on_text_camera_3 = font.render("ON", True, (0,0,0))
switch_on_rect_camera_3 = switch_on_text_camera_3.get_rect(center=(1222,600))
switch_on_rect_box_camera_3 = pygame.draw.rect(screen,(255, 255, 255),(1176,580,100,40))

## -------------OFF button set camera 3 off
switch_off_text_camera_3 = font.render("OFF", True, (0,0,0))
switch_off_rect_camera_3 = switch_off_text_camera_3.get_rect(center=(1326,600))
switch_off_rect_box_camera_3 = pygame.draw.rect(screen,(255, 255, 255),(1276,580,100,40))

# Đèn thông báo status
status_light_rect_camera_3 = pygame.Rect(1086, 590, 20, 20) 
status_light_color_camera_3 = (255,0,0)

# # --------------------------------------------Các biến kểm soát------------------------------------------------------------/

# Biến để kiểm soát
running = True
ACTIVE_AI_camera_1 = False
ACTIVE_AI_camera_2 = False
ACTIVE_AI_camera_3 = False

is_started = False

# Phần thân chính chạy app-------------------------------------------------------------------------------------------------------------------|

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            if button_start_rect.collidepoint(event.pos):
                if is_started:
                    is_started = False
                    
                    ACTIVE_AI_camera_1 = False
                    ACTIVE_AI_camera_2 = False
                    ACTIVE_AI_camera_3 = False
                    
                    button_start_color = (0,128,0)  # Màu xanh
                    button_start_text = font.render("START", True, (255, 255, 255))

                    status_light_color_camera_1 = (255,0,0)
                    status_light_color_camera_2 = (255,0,0)
                    status_light_color_camera_3 = (255,0,0)
                else:
                    is_started = True
                    
                    ACTIVE_AI_camera_1 = True
                    ACTIVE_AI_camera_2 = True
                    ACTIVE_AI_camera_3 = True
                    
                    button_start_color = (255, 0, 0)  # Màu đỏ
                    button_start_text = font.render("  END", True, (255, 255, 255))   
                    
                    status_light_color_camera_1 = (0, 255, 0)
                    status_light_color_camera_2 = (0, 255, 0)
                    status_light_color_camera_3 = (0, 255, 0)
                    
            if switch_on_rect_box_camera_1.collidepoint(event.pos):
                ACTIVE_AI_camera_1 = True
                status_light_color_camera_1 = (0, 255, 0)
            if switch_off_rect_box_camera_1.collidepoint(event.pos):
                ACTIVE_AI_camera_1 = False
                status_light_color_camera_1 = (255,0,0)
                
            if switch_on_rect_box_camera_2.collidepoint(event.pos):
                ACTIVE_AI_camera_2 = True
                status_light_color_camera_2 = (0, 255, 0)
            if switch_off_rect_box_camera_2.collidepoint(event.pos):
                ACTIVE_AI_camera_2 = False
                status_light_color_camera_2 = (255,0,0)
                
            if switch_on_rect_box_camera_3.collidepoint(event.pos):
                ACTIVE_AI_camera_3 = True
                status_light_color_camera_3 = (0, 255, 0)
            if switch_off_rect_box_camera_3.collidepoint(event.pos):
                ACTIVE_AI_camera_3 = False
                status_light_color_camera_3 = (255,0,0)
            
            # điều kiện kết thúc app    
            if  exit_clickable_area.collidepoint(event.pos):
                running = False  
                             
    # Vẽ nền trắng
    screen.fill((192,192,192))
    
    

    pygame.draw.rect(screen, button_start_color, button_start_rect, border_radius = 30)
    screen.blit(button_start_text, text_start_rect)

    screen.blit(switch_on_text_camera_1, switch_on_rect_camera_1)
    screen.blit(switch_off_text_camera_1, switch_off_rect_camera_1)
    pygame.draw.rect(screen, (0,0,0), div_ON_OFF_camera_1, 3)
    
    if ACTIVE_AI_camera_1:
        switch_on_text_camera_1 = font.render("ON", True, (255,0,0))
        switch_off_text_camera_1 = font.render("OFF", True, (0,0,0))
        pygame.draw.rect(screen, (255,0,0), switch_on_rect_box_camera_1,3)
    else:
        switch_on_text_camera_1 = font.render("ON", True, (0,0,0))
        switch_off_text_camera_1 = font.render("OFF", True, (255,0,0))
        pygame.draw.rect(screen, (255,0,0), switch_off_rect_box_camera_1,3)
    
    pygame.draw.rect(screen, status_light_color_camera_1, status_light_rect_camera_1, border_radius = 30)
    
    
    screen.blit(switch_on_text_camera_2, switch_on_rect_camera_2)
    screen.blit(switch_off_text_camera_2, switch_off_rect_camera_2)
    pygame.draw.rect(screen, (0,0,0), div_ON_OFF_camera_2, 3)
    
    if ACTIVE_AI_camera_2:
        switch_on_text_camera_2 = font.render("ON", True, (255,0,0))
        switch_off_text_camera_2 = font.render("OFF", True, (0,0,0))
        pygame.draw.rect(screen, (255,0,0), switch_on_rect_box_camera_2,3)
    else:
        switch_on_text_camera_2 = font.render("ON", True, (0,0,0))
        switch_off_text_camera_2 = font.render("OFF", True, (255,0,0))
        pygame.draw.rect(screen, (255,0,0), switch_off_rect_box_camera_2,3)
    
    pygame.draw.rect(screen, status_light_color_camera_2, status_light_rect_camera_2, border_radius = 30)
    
    
    screen.blit(switch_on_text_camera_3, switch_on_rect_camera_3)
    screen.blit(switch_off_text_camera_3, switch_off_rect_camera_3)
    pygame.draw.rect(screen, (0,0,0), div_ON_OFF_camera_3, 3)
    
    if ACTIVE_AI_camera_3:
        switch_on_text_camera_3 = font.render("ON", True, (255,0,0))
        switch_off_text_camera_3 = font.render("OFF", True, (0,0,0))
        pygame.draw.rect(screen, (255,0,0), switch_on_rect_box_camera_3,3)
    else:
        switch_on_text_camera_3 = font.render("ON", True, (0,0,0))
        switch_off_text_camera_3 = font.render("OFF", True, (255,0,0))
        pygame.draw.rect(screen, (255,0,0), switch_off_rect_box_camera_3,3)
    
    pygame.draw.rect(screen, status_light_color_camera_3, status_light_rect_camera_3, border_radius = 30)

    # Bắt đầu tính toán thời gian để đo FPS
    start_time_1 = time.time()
    ret_1, frame_1 = camera_1.read()  # Camera frame to Check bottle
    
    start_time_1 = time.time()
    ret_2, frame_2 = camera_2.read()  # Camera frame to Check water level
    
    start_time_3 = time.time()
    ret_3, frame_3 = camera_3.read()  # Camera frame to Check label

# Quản lí frame 1 ---------------------------------------------------------------------------------------         
    if ret_1:
        
        # The video uses BGR colors and PyGame needs RGB
        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
        if ACTIVE_AI_camera_1 == True:
            frame_1, ID_DEFAULT_1, ERROR_DEFAULT_1 = CHECK_BOTTLE_AI(frame_1, start_time_1)
        
        # Resize frame to show it on UI
        frame_1 = cv2.resize(frame_1, (camera_height, camera_width))
        #for some reasons the frames appeared inverted
        # The opencv camera window display on the pygame UI 
        #  is upside down compared to reality, so we have to move it
        frame_1 = cv2.flip(frame_1, 1)
        frame_1 = cv2.rotate(frame_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_1 = pygame.surfarray.make_surface(frame_1)
        
        # if (ID_DEFAULT_1 != "") and (ERROR_DEFAULT_1 != ""):
        #     id_info_error_text_1 = font.render(ID_DEFAULT_1, True, id_info_color_1)
        #     if ERROR_DEFAULT_1 == "GOOD":
        #         info_error_text_1 = font.render("GOOD", True, (0,200,0))
        #     else:
        #         info_error_text_1 = font.render("ERROR", True, (200,0,0))
        
    pygame.draw.rect(screen, (255,255,255), square_rect_1)
    pygame.draw.rect(screen, (0,0,128), square_rect_1, 3)
    screen.blit(id_title_1, id_title_1_rect)
    screen.blit(status_title_1, status_title_1_rect)
    screen.blit(id_info_error_text_1, id_info_error_rect_1)
    screen.blit(info_error_text_1, info_error_rect_1)

   
# Quản lí frame 2 ---------------------------------------------------------------------------------------
    if ret_2:
        # The video uses BGR colors and PyGame needs RGB
        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
        
        if ACTIVE_AI_camera_2 == True:
            frame_2, ID_DEFAULT_2, ERROR_DEFAULT_2 = CHECK_WATER_LEVEL_AI(frame_2, start_time_1)
            if (ID_DEFAULT_2 != "") and (ERROR_DEFAULT_2 != ""):
                id_info_error_text_2 = font.render(str(ID_DEFAULT_2), True, id_info_color_2)
                if ERROR_DEFAULT_2 == "GOOD":
                    info_error_text_2 = font.render("GOOD", True, (0,200,0))
                elif ERROR_DEFAULT_2 == "ERROR":
                    info_error_text_2 = font.render("ERROR", True, (200,0,0))
        # Resize frame to show it on UI
        frame_2 = cv2.resize(frame_2, (camera_height, camera_width))      
        #for some reasons the frames appeared inverted
        # The opencv camera window display on the pygame UI 
        #  is upside down compared to reality, so we have to move it
        frame_2 = cv2.flip(frame_2, 1)
        frame_2 = cv2.rotate(frame_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_2 = pygame.surfarray.make_surface(frame_2)


    pygame.draw.rect(screen, (255,255,255), square_rect_2)
    pygame.draw.rect(screen, (0,0,128), square_rect_2, 3)
    screen.blit(id_title_2, id_title_2_rect)
    screen.blit(status_title_2, status_title_2_rect)
    screen.blit(id_info_error_text_2, id_info_error_rect_2)
    screen.blit(info_error_text_2, info_error_rect_2)

# Quản lí frame 3 ---------------------------------------------------------------------------------------      
    if ret_3:
        # The video uses BGR colors and PyGame needs RGB
        frame_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2RGB)
        
        # Pass the frame and the AI function to output a new frame containing predictions about the object in the frame
        if ACTIVE_AI_camera_3 == True:
            frame_3, ID_DEFAULT_3, ERROR_DEFAULT_3 = CHECK_LABEL_AI(frame_3, start_time_3)
            if (ID_DEFAULT_3 != "") and (ERROR_DEFAULT_3 != ""):
                id_info_error_text_3 = font.render(str(ID_DEFAULT_3), True, id_info_color_3)
                if ERROR_DEFAULT_3 == "GOOD":
                    info_error_text_3 = font.render("GOOD", True, (0,200,0))
                elif ERROR_DEFAULT_3 == "ERROR":
                    info_error_text_3 = font.render("ERROR", True, (200,0,0))
        
        # Resize frame to show it on UI
        frame_3 = cv2.resize(frame_3, (camera_height, camera_width))
        #for some reasons the frames appeared inverted
        # The opencv camera window display on the pygame UI 
        #  is upside down compared to reality, so we have to move it
        frame_3 = cv2.flip(frame_3, 1)
        frame_3 = cv2.rotate(frame_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_3 = pygame.surfarray.make_surface(frame_3)
    
    pygame.draw.rect(screen, (255,255,255), square_rect_3)
    pygame.draw.rect(screen, (0,0,128), square_rect_3, 3)
    screen.blit(id_title_3, id_title_3_rect)
    screen.blit(status_title_3, status_title_3_rect)
    screen.blit(id_info_error_text_3, id_info_error_rect_3)
    screen.blit(info_error_text_3, info_error_rect_3)
    
#**************************************************************************************************
    screen.blit(frame_1, (10, 30))
    screen.blit(frame_2, (518, 30))
    screen.blit(frame_3, (screen_width - 510, 30))
    
    screen.blit(logo_fpt_surface,(120, screen_height - 100))
    screen.blit(exit_surface,(20, screen_height - 70))
    screen.blit(setting_surface,(20, screen_height - 120))
    
    pygame.display.flip()

camera_1.release()
camera_2.release()
camera_3.release()
pygame.quit()
