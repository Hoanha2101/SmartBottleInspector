from library import *
from AI import CHECK_BOTTLE_AI, CHECK_WATER_LEVEL_AI, CHECK_LABEL_AI, AI_COMBINE
from utils import *
from __camera__ import camera_1, camera_2, camera_3

CLEAN_CSV_BOTTLE()
CLEAN_CSV_WATER_LEVEL()
CLEAN_CSV_LABEL()

CLEAN_CSV_COMBINE()
delete_files_in_folder_image_show()

pygame.init()

screen_width = 1536
screen_height = 800
# Kích thước cửa sổ hiển thị camera
camera_height = 500
camera_width = 350

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Bottle Inspector v2.0")

font = pygame.font.Font(None, 36)
POKEFONT = pygame.font.Font("APP/font/Pokemon_GB.ttf", 18)
border_radius_button = 30


#Logo fpt ))
logo_fpt_path = os.path.join("APP/image_set/logofptuniversity.png")
logo_fpt_surface = pygame.image.load(logo_fpt_path)
logo_fpt_surface = pygame.transform.scale(logo_fpt_surface, (150, 58))

#Logo product - group ))
logo_group_path = os.path.join("APP/image_set/group.png")
logo_group_surface = pygame.image.load(logo_group_path)
logo_group_surface = pygame.transform.scale(logo_group_surface, (250, 250))

#Logo tensorRT on ))
logo_tensorRT_path = os.path.join("APP/image_set/tensorRT.png")
logo_tensorRT_surface = pygame.image.load(logo_tensorRT_path)
logo_tensorRT_surface = pygame.transform.scale(logo_tensorRT_surface, (110, 70))

#Logo tensorRT_off ))
logo_tensorRT_off_path = os.path.join("APP/image_set/tensorRT_off.png")
logo_tensorRT_off_surface = pygame.image.load(logo_tensorRT_off_path)
logo_tensorRT_off_surface = pygame.transform.scale(logo_tensorRT_off_surface, (110, 70))

#switch_on))
logo_switch_on_path = os.path.join("APP/image_set/switch_on.png")
logo_switch_on_surface = pygame.image.load(logo_switch_on_path)
logo_switch_on_surface = pygame.transform.scale(logo_switch_on_surface, (100, 90))
logo_switch_on_area = pygame.Rect(1020, screen_height - 110, 100, 90)

#switch_off))
logo_switch_off_path = os.path.join("APP/image_set/switch_off.png")
logo_switch_off_surface = pygame.image.load(logo_switch_off_path)
logo_switch_off_surface = pygame.transform.scale(logo_switch_off_surface, (100, 90))

"""Anomaly icon"""
#Logo anomaly on ))
logo_anomaly_path = os.path.join("APP/image_set/anomaly_on.png")
logo_anomaly_surface = pygame.image.load(logo_anomaly_path)
logo_anomaly_surface = pygame.transform.scale(logo_anomaly_surface, (55, 55))
#Logo anomaly_off ))
logo_anomaly_off_path = os.path.join("APP/image_set/anomaly_off.png")
logo_anomaly_off_surface = pygame.image.load(logo_anomaly_off_path)
logo_anomaly_off_surface = pygame.transform.scale(logo_anomaly_off_surface, (55, 55))
#switch_on))
anomaly_logo_switch_on_path = os.path.join("APP/image_set/switch_on.png")
anomaly_logo_switch_on_surface = pygame.image.load(anomaly_logo_switch_on_path)
anomaly_logo_switch_on_surface = pygame.transform.scale(anomaly_logo_switch_on_surface, (100, 90))
anomaly_logo_switch_on_area = pygame.Rect(820, screen_height - 110, 100, 90)
#switch_off))
anomaly_logo_switch_off_path = os.path.join("APP/image_set/switch_off.png")
anomaly_logo_switch_off_surface = pygame.image.load(anomaly_logo_switch_off_path)
anomaly_logo_switch_off_surface = pygame.transform.scale(anomaly_logo_switch_off_surface, (100, 90))


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

#exit setting button ))
exit_setting_path = os.path.join("APP/image_set/exit_setting.png")
exit_setting_surface = pygame.image.load(exit_setting_path)
exit_setting_surface = pygame.transform.scale(exit_setting_surface, (20, 20))
exit_setting_clickable_area = pygame.Rect(895, 635, 40, 40)

# hộp để setting thông tin khi bấm nút setting
big_square_setting_rect = pygame.Rect(80, 630, 850, 153)
big_square_setting_color = (255, 255, 255)

title_setting_text = POKEFONT.render("SETTINGS", True, (255, 0, 0))
title_setting_rect = title_setting_text.get_rect(center=(170, 715))

title_combine_text = POKEFONT.render("Combine", True, (0, 0, 0))
title_combine_rect = title_combine_text.get_rect(center=(400, 670))

#module_3_off))
module_3_off_path = os.path.join("APP/image_set/module_3_off.png")
module_3_off_surface = pygame.image.load(module_3_off_path)
module_3_off_surface = pygame.transform.scale(module_3_off_surface, (40, 40))
module_3_off_area = pygame.Rect(550, 650, 40, 40)

#module_3_on))
module_3_on_path = os.path.join("APP/image_set/module_3_on.png")
module_3_on_surface = pygame.image.load(module_3_on_path)
module_3_on_surface = pygame.transform.scale(module_3_on_surface, (40, 40))
module_3_on_area = pygame.Rect(625, 625, 40, 40)

#switch_on_combine))
logo_switch_on_combine_path = os.path.join("APP/image_set/switch_on.png")
logo_switch_on_combine_surface = pygame.image.load(logo_switch_on_combine_path)
logo_switch_on_combine_surface = pygame.transform.scale(logo_switch_on_combine_surface, (100, 90))
logo_switch_on_combine_area = pygame.Rect(625, 625, 100, 90)


#switch_off_combine))
logo_switch_off_combine_path = os.path.join("APP/image_set/switch_off.png")
logo_switch_off_combine_surface = pygame.image.load(logo_switch_off_combine_path)
logo_switch_off_combine_surface = pygame.transform.scale(logo_switch_off_combine_surface, (100, 90))

#module_combine_off))
module_combine_off_path = os.path.join("APP/image_set/module_combine_off.png")
module_combine_off_surface = pygame.image.load(module_combine_off_path)
module_combine_off_surface = pygame.transform.scale(module_combine_off_surface, (40, 40))
module_combine_off_area = pygame.Rect(750, 650, 40, 40)

#module_combine_on))
module_combine_on_path = os.path.join("APP/image_set/module_combine_on.png")
module_combine_on_surface = pygame.image.load(module_combine_on_path)
module_combine_on_surface = pygame.transform.scale(module_combine_on_surface, (40, 40))
module_combine_on_area = pygame.Rect(750, 650, 40, 40)

<<<<<<< HEAD
=======
ANNO_text = POKEFONT.render("ANNO", True, (0, 0, 0))
ANNO_rect = ANNO_text.get_rect(center=(400, 742))

#switch_on_annotation))
logo_switch_on_ANNO_path = os.path.join("APP/image_set/switch_on.png")
logo_switch_on_ANNO_surface = pygame.image.load(logo_switch_on_ANNO_path)
logo_switch_on_ANNO_surface = pygame.transform.scale(logo_switch_on_ANNO_surface, (100, 90))
logo_switch_on_ANNO_area = pygame.Rect(430, 690, 100, 90)

#switch_off_annotation))
logo_switch_off_ANNO_path = os.path.join("APP/image_set/switch_off.png")
logo_switch_off_ANNO_surface = pygame.image.load(logo_switch_off_ANNO_path)
logo_switch_off_ANNO_surface = pygame.transform.scale(logo_switch_off_ANNO_surface, (100, 90))

"--------------------------------------------------"
DEMO_text = POKEFONT.render("DEMO", True, (0, 0, 0))
DEMO_rect = DEMO_text.get_rect(center=(660, 742))

# Button demo 1
demo_1_rect = pygame.Rect(740, 720, 40, 40)  
demo_1_color = (128,0,0)
demo_1_text = font.render("1", True, (255, 255, 255))
text_demo_1_rect = demo_1_text.get_rect(center=demo_1_rect.center)

# Button demo 2
demo_2_rect = pygame.Rect(790, 720, 40, 40)  
demo_2_color = (128,0,0)
demo_2_text = font.render("2", True, (255, 255, 255))
text_demo_2_rect = demo_2_text.get_rect(center=demo_2_rect.center)

# Button demo 3
demo_3_rect = pygame.Rect(840, 720, 40, 40)  
demo_3_color = (128,0,0)
demo_3_text = font.render("3", True, (255, 255, 255))
text_demo_3_rect = demo_3_text.get_rect(center=demo_3_rect.center)

"--------------------------------------------------"

>>>>>>> UI-Hoan
# Button Start - End
button_start_rect = pygame.Rect(1385, 700, 120, 50)  
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


'''
    ========================================show information - combine=========================================================
'''
square_rect_COMBINE= pygame.Rect(900, 30, 606, 550) 
# Tạo đường phân tách hộp thông tin
separation_rect = pygame.Rect(903, 400, 600, 3) 
separation_color = (0, 0, 0) 

# Đèn thông báo status
status_light_rect_combine = pygame.Rect(70, 590, 20, 20) 
status_light_color_combine = (255,0,0)

# Các thành phần trong hộp thông tin combine
font_ = pygame.font.Font(None, 40)

id_error_text_combine = font_.render("ID : ", True, (0, 0, 0))
id_error_rect_combine = id_error_text_combine.get_rect(center=(1054, 455))

status_text_combine = font_.render("STATUS : ", True, (0, 0, 0))
status_rect_combine = status_text_combine.get_rect(center=(1020, 520))

# Thông tin good - error
id_info_color_combine = (0,200,0)
id_info_error_text_combine = font_.render("-", True, id_info_color_combine)
id_info_error_rect_combine = id_info_error_text_combine.get_rect(center=(1200, 455))

status_info_color_combine = (0,200,0)
status_info_error_text_combine = font_.render("-", True, status_info_color_combine)
status_info_error_rect_combine = status_info_error_text_combine.get_rect(center=(1200, 520))

# # --------------------------------------------Các biến kểm soát------------------------------------------------------------/

ID_DEFAULT_COMBINE = None

# Biến để kiểm soát
running = True
ACTIVE_AI_camera_1 = False
ACTIVE_AI_camera_2 = False
ACTIVE_AI_camera_3 = False
ACTIVE_AI_camera_COMBINE = False

is_tensorRT = False
is_anomaly = False
is_started = False

activate_optimize_RT = False
is_square_setting_visible = False
is_combine_module = False
is_setting_icon = True
<<<<<<< HEAD
=======
is_ANNO = False

is_demo_module = 1

>>>>>>> UI-Hoan
# Phần thân chính chạy app-------------------------------------------------------------------------------------------------------------------|

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            if button_start_rect.collidepoint(event.pos):
                if is_square_setting_visible == False:
                    if is_started:
                        if is_combine_module == False:
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
                            is_started = False
                            ACTIVE_AI_camera_COMBINE = False
                            button_start_color = (0,128,0)  # Màu xanh
                            button_start_text = font.render("START", True, (255, 255, 255))
                            status_light_color_combine = (255, 0, 0)
                            
                    else:
                        if is_combine_module == False:
                            is_started = True
                            
                            ACTIVE_AI_camera_1 = True
                            ACTIVE_AI_camera_2 = True
                            ACTIVE_AI_camera_3 = True
                            
                            button_start_color = (255, 0, 0)  # Màu đỏ
                            button_start_text = font.render("  END", True, (255, 255, 255))   
                            
                            status_light_color_camera_1 = (0, 255, 0)
                            status_light_color_camera_2 = (0, 255, 0)
                            status_light_color_camera_3 = (0, 255, 0)
                            
                        else:
                            is_started = True
                            ACTIVE_AI_camera_COMBINE = True
                            button_start_color = (255, 0, 0)  # Màu đỏ
                            button_start_text = font.render("  END", True, (255, 255, 255))
                            status_light_color_combine = (0, 255, 0)
<<<<<<< HEAD
                            
                    
=======
            
            if is_square_setting_visible == True:
                if demo_1_rect.collidepoint(event.pos):
                    is_demo_module = 1
                if demo_2_rect.collidepoint(event.pos):
                    is_demo_module = 2
                if demo_3_rect.collidepoint(event.pos):
                    is_demo_module = 3
                
                if logo_switch_on_ANNO_area.collidepoint(event.pos):
                    is_ANNO = not is_ANNO
                
>>>>>>> UI-Hoan
            if is_combine_module == False:
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
            
            if exit_setting_clickable_area.collidepoint(event.pos):
                if is_square_setting_visible == True:
                    is_square_setting_visible = not is_square_setting_visible
            
            if anomaly_logo_switch_on_area.collidepoint(event.pos):
                if (is_combine_module == True) and (is_square_setting_visible == False):
                    is_anomaly = not is_anomaly
              
            if setting_clickable_area.collidepoint(event.pos):
                if is_started == False:
                    is_square_setting_visible = not is_square_setting_visible   
                    
            if logo_switch_on_combine_area.collidepoint(event.pos):   
                if  is_square_setting_visible == True:
                    is_combine_module = not is_combine_module

                    
            # điều kiện kết thúc app    
            if  exit_clickable_area.collidepoint(event.pos):
                running = False  
                
            # điều kiện kết thúc app    
            if  logo_switch_on_area.collidepoint(event.pos):
                is_tensorRT = not is_tensorRT
                activate_optimize_RT = not activate_optimize_RT
                
    # Vẽ nền trắng
    screen.fill((192,192,192))
    if is_square_setting_visible == False:
        pygame.draw.rect(screen, button_start_color, button_start_rect, border_radius = 30)
        screen.blit(button_start_text, text_start_rect)
        
        
    if is_combine_module == False:
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

<<<<<<< HEAD
        # Bắt đầu tính toán thời gian để đo FPS
        start_time_1 = time.time()
        ret_1, frame_1 = camera_1.read()  # Camera frame to Check bottle
        
        start_time_1 = time.time()
        ret_2, frame_2 = camera_2.read()  # Camera frame to Check water level
        
        start_time_3 = time.time()
        ret_3, frame_3 = camera_3.read()  # Camera frame to Check label
=======
        if is_demo_module == 1:
            # Bắt đầu tính toán thời gian để đo FPS
            start_time_1 = time.time()
            ret_1, frame_1 = camera_1.read()  # Camera frame to Check bottle
            
            start_time_1 = time.time()
            ret_2, frame_2 = camera_2.read()  # Camera frame to Check water level
            
            start_time_3 = time.time()
            ret_3, frame_3 = camera_3.read()  # Camera frame to Check label
            
        if is_demo_module == 2:
            
            start_time_1 = time.time()
            ret_1, frame_1 = camera_2.read()
            
            start_time_1 = time.time()
            ret_2, frame_2 = camera_1.read() 
            
            start_time_3 = time.time()
            ret_3, frame_3 = camera_3.read()  
            
        if is_demo_module == 3:
            
            start_time_1 = time.time()
            ret_1, frame_1 = camera_2.read()
            
            start_time_1 = time.time()
            ret_2, frame_2 = camera_3.read()
            
            start_time_3 = time.time()
            ret_3, frame_3 = camera_1.read()
>>>>>>> UI-Hoan

    # Quản lí frame 1 ---------------------------------------------------------------------------------------         
        if ret_1:
            
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
            
            if ACTIVE_AI_camera_1 == True:
                
<<<<<<< HEAD
                frame_1, ID_DEFAULT_1, ERROR_DEFAULT_1 = CHECK_BOTTLE_AI(frame_1, start_time_1, activate_optimize_RT)
=======
                frame_1, ID_DEFAULT_1, ERROR_DEFAULT_1 = CHECK_BOTTLE_AI(frame_1, start_time_1,is_ANNO , activate_optimize_RT)
>>>>>>> UI-Hoan
                
                if (ID_DEFAULT_1 != "") and (ERROR_DEFAULT_1 != ""):
                    id_info_error_text_1 = font.render(str(ID_DEFAULT_1), True, id_info_color_1)
                    if ERROR_DEFAULT_1 == "GOOD":
                        info_error_text_1 = font.render("GOOD", True, (0,200,0))
                    elif ERROR_DEFAULT_1 == "ERROR":
                        info_error_text_1 = font.render("ERROR", True, (200,0,0))
            # Resize frame to show it on UI
            frame_1 = cv2.resize(frame_1, (camera_height, camera_width))      
            #for some reasons the frames appeared inverted
            # The opencv camera window display on the pygame UI 
            #  is upside down compared to reality, so we have to move it
            frame_1 = cv2.flip(frame_1, 1)
            frame_1 = cv2.rotate(frame_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_1 = pygame.surfarray.make_surface(frame_1)
            
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
<<<<<<< HEAD
                frame_2, ID_DEFAULT_2, ERROR_DEFAULT_2 = CHECK_WATER_LEVEL_AI(frame_2, start_time_1)
=======
                frame_2, ID_DEFAULT_2, ERROR_DEFAULT_2 = CHECK_WATER_LEVEL_AI(frame_2, start_time_1, is_ANNO)
>>>>>>> UI-Hoan
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
<<<<<<< HEAD
                frame_3, ID_DEFAULT_3, ERROR_DEFAULT_3 = CHECK_LABEL_AI(frame_3, start_time_3, activate_optimize_RT)
=======
                frame_3, ID_DEFAULT_3, ERROR_DEFAULT_3 = CHECK_LABEL_AI(frame_3, start_time_3, is_ANNO, activate_optimize_RT)
>>>>>>> UI-Hoan
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
          
    """Combine 3 module""" 
    if is_combine_module == True:  

        pygame.draw.rect(screen, status_light_color_combine, status_light_rect_combine, border_radius = 30)
        
        # Bắt đầu tính toán thời gian để đo FPS
        start_time_COMBINE = time.time()
        ret_COMBINE , frame_COMBINE  = camera_1.read()
        
        if ret_COMBINE:
            
            frame_COMBINE  = cv2.cvtColor(frame_COMBINE , cv2.COLOR_BGR2RGB)
            if ACTIVE_AI_camera_COMBINE  == True: 
                frame_COMBINE , ID_DEFAULT_COMBINE , ERROR_DEFAULT_COMBINE  = AI_COMBINE(frame_COMBINE , start_time_COMBINE , activate_optimize_RT)
                
                if (ID_DEFAULT_COMBINE  != "") and (ERROR_DEFAULT_COMBINE  != ""):
                    id_info_error_text_combine  = font.render(str(ID_DEFAULT_COMBINE ), True, (200,0,0))
                    if ERROR_DEFAULT_COMBINE  == "GOOD":
                        status_info_error_text_combine  = font.render("GOOD", True, (0,200,0))
                    elif ERROR_DEFAULT_COMBINE  == "ERROR":
                        status_info_error_text_combine  = font.render("ERROR", True, (200,0,0))
            
            # Resize frame to show it on UI
            frame_COMBINE  = cv2.resize(frame_COMBINE , (800, 550))
            #for some reasons the frames appeared inverted
            # The opencv camera window display on the pygame UI 
            #  is upside down compared to reality, so we have to move it
            frame_COMBINE  = cv2.flip(frame_COMBINE , 1)
            frame_COMBINE  = cv2.rotate(frame_COMBINE , cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_COMBINE  = pygame.surfarray.make_surface(frame_COMBINE )
            
        pygame.draw.rect(screen, (255,255,255), square_rect_COMBINE )
        pygame.draw.rect(screen, (0,0,128), square_rect_COMBINE , 3)
        
        pygame.draw.rect(screen, (0,0,0), separation_rect, 3)
        pygame.draw.rect(screen, separation_color, separation_rect)
        
        screen.blit(frame_COMBINE , (30, 30))

        screen.blit(id_error_text_combine, id_error_rect_combine)
        screen.blit(status_text_combine, status_rect_combine)

        screen.blit(id_info_error_text_combine, id_info_error_rect_combine)
        screen.blit(status_info_error_text_combine, status_info_error_rect_combine)
        
        if ID_DEFAULT_COMBINE != None:      
            name_image_write = "APP/image_show/" + str(ID_DEFAULT_COMBINE) + ".jpg"
            if os.path.exists(name_image_write):
                #image Show
                captured_image_surface = pygame.image.load(os.path.join(name_image_write))
                # captured_image_surface = pygame.transform.scale(captured_image_surface, (390, 300))
                screen.blit(captured_image_surface,(screen_width - 390, 62))
            else:
                pass
        
#----------------------------------------------------------------- 
     
    if is_tensorRT == True:
        screen.blit(logo_tensorRT_surface,(1100, screen_height - 105))
        screen.blit(logo_switch_on_surface,(1020, screen_height - 110))
    else:   
        screen.blit(logo_tensorRT_off_surface,(1100, screen_height - 105))
        screen.blit(logo_switch_off_surface,(1020, screen_height - 110))
    
    if is_combine_module:
        if is_anomaly == True:
            screen.blit(logo_anomaly_surface,(900, screen_height - 102))
            screen.blit(anomaly_logo_switch_on_surface,(820, screen_height - 110))
        else:   
            screen.blit(logo_anomaly_off_surface,(900, screen_height - 102))
            screen.blit(anomaly_logo_switch_off_surface,(820, screen_height - 110))
    
    screen.blit(exit_surface,(20, screen_height - 70))
    
    screen.blit(logo_fpt_surface,(120, screen_height - 100))
    screen.blit(logo_group_surface,(280, screen_height - 190))
    
    if is_started == False:
        screen.blit(setting_surface,(20, screen_height - 120))
    if is_square_setting_visible:
        pygame.draw.rect(screen, big_square_setting_color, big_square_setting_rect)
        pygame.draw.rect(screen, (100,0,200), big_square_setting_rect, 3)
        
        screen.blit(exit_setting_surface,(903, 635))
        
        screen.blit(title_setting_text, title_setting_rect)
        screen.blit(title_combine_text, title_combine_rect)
        
<<<<<<< HEAD
=======
        screen.blit(DEMO_text, DEMO_rect)
        
        screen.blit(ANNO_text, ANNO_rect)
        
        if is_demo_module == 1:
            demo_1_color = (0,128,0)
            demo_2_color = (128,0,0)
            demo_3_color = (128,0,0)
            
        if is_demo_module == 2:
            demo_1_color = (128,0,0)
            demo_2_color = (0,128,0)
            demo_3_color = (128,0,0)
            
        if is_demo_module == 3:
            demo_1_color = (128,0,0)
            demo_2_color = (128,0,0)
            demo_3_color = (0,128,0)
            
        pygame.draw.rect(screen, demo_1_color, demo_1_rect, border_radius= 30)
        screen.blit(demo_1_text, text_demo_1_rect)
        
        pygame.draw.rect(screen, demo_2_color, demo_2_rect, border_radius= 30)
        screen.blit(demo_2_text, text_demo_2_rect)
        
        pygame.draw.rect(screen, demo_3_color, demo_3_rect, border_radius= 30)
        screen.blit(demo_3_text, text_demo_3_rect)
        
        if is_ANNO:
            screen.blit(logo_switch_on_ANNO_surface,(430, 690))
        else:
            screen.blit(logo_switch_off_ANNO_surface,(430, 690))
        
>>>>>>> UI-Hoan
        if is_combine_module == True:
            screen.blit(module_3_off_surface,(550, 650))
            screen.blit(module_combine_on_surface,(750, 650))
            screen.blit(logo_switch_on_combine_surface,(625, 625))
        else:
            screen.blit(module_3_on_surface,(550, 650))
            screen.blit(module_combine_off_surface,(750, 650))
            screen.blit(logo_switch_off_combine_surface,(625, 625))
            
    pygame.display.flip()

camera_1.release()
camera_2.release()
camera_3.release()
pygame.quit()
<<<<<<< HEAD
=======

>>>>>>> UI-Hoan
