from tkinter import *

#dummy values
ra = 12.34  # Right Ascension
dec = 20.56  # Declination
roll = 5.78  # Roll angle
q = [0.9966, 0.0044, 0, 0]

def presentoutput(ra,dec,roll,q):
    ra = round(ra, 3)
    dec = round(dec, 3)
    roll = round(roll, 3)
    
    rounded_array = []
    for number in q:
        rounded_array.append(round(number, 3))
    
    q = rounded_array

    #main window
    m = Tk()
    m.geometry("1024x700") #sizeof the main window
    m.title("Attitude Calculation")

    # Styling
    font_size = 15
    font_family = "Arial"
    Font = (font_family, font_size)

    bg_color = "lightgray"
    fg_color = "black"
    label_width = 20
    label_padx = 5
    label_pady = 5

    intro_label = Label(m, text="Attitude Calulation",width=66, font=Font)
    intro_label.pack()

    #to display the image after centroiding
    canvas = Canvas(m, width = 720, height = 480)
    canvas.pack()
    img = PhotoImage(file="test69.png") #file path
    canvas.create_image(20,20, anchor=NW, image=img)

    gap_label = Label(m, text="")  # Empty label for space
    gap_label.pack(pady=10)

    #construct a frame
    main_frame = LabelFrame(m)
    main_frame.pack(side='top')

    ra_frame = LabelFrame(main_frame, text="Right Ascension (RA):", bg=bg_color, fg=fg_color, padx=label_padx, pady=label_pady)
    ra_frame.pack(side='left')
    ra_value_label = Label(ra_frame, text=str(ra)+" deg", width=label_width, bg=bg_color, fg=fg_color)  # Displays RA value
    ra_value_label.pack()

    dec_frame = LabelFrame(main_frame, text="Declination (DEC):", bg=bg_color, fg=fg_color, padx=label_padx, pady=label_pady)
    dec_frame.pack(side='left')
    dec_value_label = Label(dec_frame, text=str(dec)+" deg", width=label_width, bg=bg_color, fg=fg_color)  # Displays DEC value
    dec_value_label.pack()

    roll_frame = LabelFrame(main_frame, text="Roll (ROLL):", bg=bg_color, fg=fg_color, padx=label_padx, pady=label_pady)
    roll_frame.pack(side='left')
    roll_value_label = Label(roll_frame, text=str(roll)+" deg", width=label_width, bg=bg_color, fg=fg_color)  # Displays Roll value
    roll_value_label.pack()

    # Create label for quaternion
    q_frame = LabelFrame(m, text="Quaternion (q):", bg=bg_color, fg=fg_color, padx=label_padx, pady=label_pady)
    q_frame.pack()
    q_label = Label(q_frame, text=str(q),width=66, bg=bg_color, fg=fg_color)  # Update with calculated quaternion
    q_label.pack(fill="x")

    m.mainloop()
