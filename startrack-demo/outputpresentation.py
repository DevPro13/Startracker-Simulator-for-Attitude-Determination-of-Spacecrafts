from tkinter import *

#dummy values
ra = 12.34  # Right Ascension
dec = 20.56  # Declination
roll = 5.78  # Roll angle
q = [0.9966, 0.0044, 0, 0]

#main window
m = Tk()
m.geometry("1024x768") #sizeof the main window

#to display the image after centroiding
canvas = Canvas(m, width = 720, height = 480)
canvas.pack()
img = PhotoImage(file="test69.png") #file path
canvas.create_image(20,20, anchor=NW, image=img)

intro_label = Label(m, text="Attitude Calulation",width=66)
intro_label.pack()

#construct a frame
main_frame = LabelFrame(m)
main_frame.pack(side='top')

ra_frame = LabelFrame(main_frame, text="Right Ascension (RA):", bg="lightgray",padx=5, pady=5)
ra_frame.pack(side='left')
ra_value_label = Label(ra_frame, text=str(ra), width=20)  # Displays RA value
ra_value_label.pack()

dec_frame = LabelFrame(main_frame, text="Declination (DEC):", bg="lightgray",padx=5, pady=5)
dec_frame.pack(side='left')
dec_value_label = Label(dec_frame, text=str(dec),width=20)  # Displays DEC value
dec_value_label.pack()

roll_frame = LabelFrame(main_frame, text="Roll (ROLL):", bg="lightgray", padx=5, pady=5)
roll_frame.pack(side='left')
roll_value_label = Label(roll_frame, text=str(roll),width=20)  # Displays Roll value
roll_value_label.pack()

# Create label for quaternion
q_frame = LabelFrame(m, text="Quaternion (q):", bg="lightgray", padx=5, pady=5)
q_frame.pack()
q_label = Label(q_frame, text=str(q),width=66)  # Update with calculated quaternion
q_label.pack()

m.mainloop()
