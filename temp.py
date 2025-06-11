# import gym
# from env import EnvWrapper
# env=EnvWrapper(True)

# done,info,_=env.reset()
# i=0
# total_reward=0
# while True:
#     action=env.sample_action()
#     state,reward,done,info,_=env.step(action)
#     i+=1
#     if done :
#         break
# print(i)
import gym
import cv2
import numpy as np
def resize_image(img: np.ndarray, target_size: tuple, interpolation=cv2.INTER_AREA) -> np.ndarray:
    """
    高效缩放单张图像（h, w, 3），返回缩放后的图像。

    参数：
        img: np.ndarray，输入图像（h, w, 3）
        target_size: tuple，目标大小 (new_width, new_height)
        interpolation: 插值方式，默认使用 INTER_AREA（适合缩小）
    
    返回：
        缩放后的图像 (new_height, new_width, 3)
    """
    resized = cv2.resize(img, target_size, interpolation=interpolation)
    return resized
env = gym.make("ALE/MontezumaRevenge-v5",render_mode='rgb_array')
obs,_=env.reset()
obs=cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
obs=resize_image(obs[20:198,:,:],(160,160))
cv2.imwrite('img2.png',obs)
# done,info=env.reset()
# i=0
# color_dict={}
# import numpy as np
# # from scipy.signal import correlate2d

# # def detect_agent_by_template(image_rgb, template_rgb):
# #     """
# #     用模板匹配的方法检测 agent 位置。
    
# #     参数:
# #         image_rgb: (H, W, 3) RGB 图像
# #         template_rgb: (h, w, 3) RGB 模板图像

# #     返回:
# #         (x, y): agent 的位置（模板左上角）
# #     """
# #     # 对 RGB 每个通道进行归一化的 cross-correlation
# #     responses = []
# #     for c in range(3):
# #         img_c = image_rgb[:, :, c].astype(np.float32)
# #         tpl_c = template_rgb[:, :, c].astype(np.float32)
# #         response = correlate2d(img_c, tpl_c[::-1, ::-1], mode='valid')
# #         responses.append(response)

# #     # 总响应图（三个通道加权平均）
# #     total_response = sum(responses)
    
# #     # 找到最大响应点
# #     y, x = np.unravel_index(np.argmax(total_response), total_response.shape)
# #     return (x, y), total_response

# while True:
#     # action=env.action_space.sample()
#     # state,reward,done,truct,info=env.step(action)
#     # img=state
#     img=cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
#     img=resize_image(img[20:198,:,:],(160,160))
#     # lower_bound=np.array([60,60,190])
#     # high_bound=np.array([80,80,215])
#     # img_mask=np.all((img>=lower_bound)&(img<=high_bound),axis=-1)
#     # img_mask=img_mask.astype(np.float32)
#     # img=img.astype(np.float32)
#     # img=img_mask[...,None]*img
#     # img=img.astype(np.uint8)
#     # img=img[53:55,78:82,:]
#     # h,w,_=img.shape
#     # for i in range(h):
#     #     for j in range(w):
#     #         name=str(int(img[i,j,0]))+' '+str(int(img[i,j,1]))+' '+str(int(img[i,j,2]))
#     #         color_dict[name]=1
#     # print(color_dict)
#     cv2.imwrite('template.png',img[48:65,76:84,:])
#     break
#     # i+=1
#     # if i==50:
#     #     break
#     #72 72 200
# from moviepy.editor import VideoFileClip
# import numpy as np

# def calculate_frame_means(video_path):
#     """
#     使用moviepy计算视频每一帧的像素均值（RGB三通道分别计算）
    
#     参数:
#         video_path (str): 视频文件路径
        
#     返回:
#         list: 包含每一帧均值结果的列表，每个元素是一个包含RGB三通道均值的元组
#     """
#     # 加载视频文件
#     clip = VideoFileClip(video_path)
    
#     frame_means = []
    
#     # 逐帧处理
#     for i, frame in enumerate(clip.iter_frames()):
#         # 计算当前帧的RGB三通道均值
#         mean_r = np.mean(frame[:, :, 0])  # 红色通道
#         mean_g = np.mean(frame[:, :, 1])  # 绿色通道
#         mean_b = np.mean(frame[:, :, 2])  # 蓝色通道
        
#         frame_means.append((mean_r, mean_g, mean_b))
        
#         # 可选：打印进度
#         if (i + 1) % 30 == 0:  # 每30帧打印一次
#             print(f"已处理 {i + 1} 帧...")
    
#     # 关闭视频文件
#     clip.close()
    
#     print(f"视频处理完成，共处理 {len(frame_means)} 帧")
#     return frame_means

# if __name__ == "__main__":
#     # 替换为你的视频文件路径
#     video_file = "/home/cw/MontezumaRevenge/videos/2/5.mp4"
    
#     # 计算帧均值
#     means = calculate_frame_means(video_file)
    
#     if means:
#         # 打印前5帧的结果作为示例
#         print("\n前5帧的像素均值（R, G, B）：")
#         for i in range(min(5, len(means))):
#             print(f"帧 {i+1}: {means[i]}")
        
#         # 可选：保存结果到文件
#         with open("frame_means_moviepy.txt", "w") as f:
#             for i, (r, g, b) in enumerate(means):
#                 f.write(f"帧 {i+1}: R={r:.2f}, G={g:.2f}, B={b:.2f}\n")
#         print("结果已保存到 frame_means_moviepy.txt")
# a={}
# a[1]=1
# a[2]=2
# print(a)
# import numpy as np
# import time
# import torch
# x=np.random.rand(1, 100,100)
# start_time = time.time()
# x=torch.tensor(x,dtype=torch.float32).to('cuda:4')
# # y=x.mean(axis=(1,2))
# end_time = time.time()
# print("Mean calculation time:", (end_time - start_time)*1000, "ms")
# import gym
# env=gym.make("ALE/MontezumaRevenge-v5",render_mode='rgb_array')
# env.reset()
# ale = env.unwrapped.ale 
# ram = ale.getRAM()  
# print(ram)
# import numpy as np
# obs = [1, 2, 3]
# history_len = 3
# history_obs = [obs for _ in range(history_len)]  # 3个对同一列表的引用
# x = np.array(history_obs)  # 数组中仍然是引用

# x[0][0] = 99  # 修改原始obs
# print(x)  # 会看到数组中所有第一个元素都变成了99
# import gym
# import pygame
# import numpy as np

# # 初始化环境
# env = gym.make("ALE/MontezumaRevenge-v5", render_mode="rgb_array")
# obs, _ = env.reset()

# # 初始化 pygame 用于键盘控制
# pygame.init()
# screen = pygame.display.set_mode((640, 420))
# pygame.display.set_caption("MontezumaRevenge RAM Inspector")

# clock = pygame.time.Clock()

# # 动作映射（可扩展）
# # 你可以通过打印 env.unwrapped.get_action_meanings() 获取动作含义
# action_map = {
#     pygame.K_LEFT: 3,   # LEFT
#     pygame.K_RIGHT: 2,  # RIGHT
#     pygame.K_UP: 12,    # UP
#     pygame.K_DOWN: 13,  # DOWN
#     pygame.K_SPACE: 1,  # FIRE
#     pygame.K_a: 14,     # UPRIGHT
#     pygame.K_s: 15,     # UPRIGHT + FIRE
#     pygame.K_d: 16,     # UPLEFT
# }

# running = True
# action = 0

# def show_ram_diff(ram_before, ram_after):
#     diff = [(i, b, a) for i, (b, a) in enumerate(zip(ram_before, ram_after)) if b != a]
#     print("Changed RAM indices:")
#     for i, before, after in diff:
#         print(f"  RAM[{i:3}] : {before:3} -> {after:3}")
#     print()

# while running:
#     ram_before = env.unwrapped.ale.getRAM()
#     obs, _, terminated, truncated, _ = env.step(action)
#     ram_after = env.unwrapped.ale.getRAM()

#     # 显示图像
#     rgb = env.render()
#     surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
#     surf = pygame.transform.scale(surf, (640, 420))
#     screen.blit(surf, (0, 0))
#     pygame.display.flip()

#     # 打印 RAM 差异
#     show_ram_diff(ram_before, ram_after)

#     action = 0  # 默认 no-op
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key in action_map:
#                 action = action_map[event.key]
#             elif event.key == pygame.K_r:
#                 obs, _ = env.reset()
#             elif event.key == pygame.K_ESCAPE:
#                 running = False

#     if terminated or truncated:
#         obs, _ = env.reset()

#     clock.tick(5)  # 控制速度

# env.close()
# pygame.quit()
