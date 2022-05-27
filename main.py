from myAgent import VacuumAgent, vacuumAgentProgram
from myEnv import myEnv
from tkinter import *

def main():
    root = Tk()
    root.title("Vacuum Environment")
    root.geometry("540x560")
    root.resizable(0, 0)
    frame = Frame(root, bg='black')

    frame.pack(side='bottom')
    run_button = Button(frame, text='Run', height=2,
                         width=6, padx=2, pady=2)
    run_button.pack(side='left')
    frame.pack(side='bottom')
    env = myEnv(root, width=10, height=10)
    #agt = XYReflexAgent(program=XYReflexAgentProgram)
    #init_state = ...
    agt_program = vacuumAgentProgram()
    agt = VacuumAgent(program=agt_program)

    env.add_thing(agt, location=(1, 1))
    run_button.config(command=env.run_env)

    root.mainloop()


if __name__ == '__main__':
    main()

