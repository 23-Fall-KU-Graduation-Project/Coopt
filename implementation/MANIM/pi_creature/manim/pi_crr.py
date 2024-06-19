from manimlib.imports import *
from manimlib.for_3b1b_videos.pi_creature_animations import *
from manimlib.for_3b1b_videos.pi_creature import *
from manimlib.for_3b1b_videos.pi_class import *
from manimlib.for_3b1b_videos.pi_creature_scene import *

class NowWeHaveEmotions(TeacherStudentsScene):
    def construct(self):
        self.change_student_modes('happy', 'hooray', 'well')
        self.play(self.teacher.change_mode, 'happy')
        self.teacher_says("Now we have emotions!")
        self.wait()
        self.student_says("Hooray!", student_index=1, target_mode='hooray',
                          added_anims=[self.teacher.change, "hooray"])
        self.play(self.teacher.change_mode, 'hooray')
        self.wait(2)
        self.play(RemovePiCreatureBubble(self.students[1]), self.students[1].change_mode, 'hooray')
        self.wait(3)