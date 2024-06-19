from manimlib.imports import *
from manimlib.for_3b1b_videos.pi_creature_animations import *
from manimlib.for_3b1b_videos.pi_creature import *
from manimlib.for_3b1b_videos.pi_class import *
from manimlib.for_3b1b_videos.pi_creature_scene import *

class WhatTheMeaning(TeacherStudentsScene):
    def construct(self):
        self.change_student_modes('confused', 'hesitant', 'maybe')

        self.student_says("So what's the meaning of that expressions?", student_index=1, target_mode='hesitant')
        self.play(self.teacher.change_mode, 'happy')
        self.wait(1)
        self.play(RemovePiCreatureBubble(self.students[1]))
        self.teacher_says("Let's look at some visualizations to make it clear.")
        self.wait(2)
        self.student_says("That's awesome!", student_index=1, target_mode='hooray',
                          added_anims=[self.teacher.change, "hooray"])
        self.play(self.teacher.change_mode, 'hooray')
        self.wait(2)
        self.play(RemovePiCreatureBubble(self.students[1]), self.students[1].change_mode, 'hooray')
        self.wait(3)