from shot_video import *
import os
import time
from datetime import datetime



if __name__ == "__main__":
    reel_gen = ReelGenerator()
    story = "在一片广袤的科技星空下，AWS如一柄闪耀着银色光芒的利剑静静悬浮。这把利剑的剑身流转着云计算的灵动数据流，剑锋锐利如同切割黎明的第一缕阳光。当我握住剑柄的那一刻，数字化转型的荆棘丛生之路顿时豁然开朗，如同劈开浓雾见晴天。利剑所指之处，道路两旁绽放出创新的繁花，照亮了企业腾飞的征程，恰似黎明前升起的启明星指引着前行的方向。"
    shots = generate_shots(reel_gen, story)
    print(shots)
    image_files = generate_shot_image(reel_gen, shots)
    reel_prompts = generate_reel_prompts(reel_gen, shots,image_files)
    video_files = generate_shot_vidoes(reel_gen, image_files, reel_prompts)
    final_video = sistch_vidoes(reel_gen, video_files)
