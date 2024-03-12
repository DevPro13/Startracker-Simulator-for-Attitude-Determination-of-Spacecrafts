from tkinter import *
from PIL import Image, ImageTk
import os
import scaleimage

#dummy values
ra = 12.34  # Right Ascension
dec = 20.56  # Declination
roll = 5.78  # Roll angle
q = [0.9966, 0.0044, 0, 0]

def presentoutput(impath, ra, dec, roll, q):
    ra = round(ra, 3)
    dec = round(dec, 3)
    roll = round(roll, 3)
    rounded_array = []
    for number in q:
        rounded_array.append(round(number, 3))
    q = rounded_array
    # Split the path into components (directory, filename, extension)
    head, tail = os.path.split(impath)
    filename, extension = os.path.splitext(tail)

    # Move one directory behind (assuming the image folder exists one level above)
    new_head = os.path.dirname(head)

    # Create the new path with modified filename
    ann_impath = os.path.join(new_head, f"ann_{filename}{extension}")
    pot_impath = os.path.join(new_head, f"pot_{filename}{extension}")

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
    subIntro_label = Label(m, text=filename,width=66, font=Font)
    subIntro_label.pack()

    # Load images using PIL
    image1 = Image.open(pot_impath)
    image2 = Image.open(ann_impath)

    #resizing so that it fits into the canvas
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate new width while maintaining aspect ratio
    new_width = min(400, width1, width2)
    new_height1 = int(height1 * (new_width / width1))
    new_height2 = int(height2 * (new_width / width2))

    new_image1 = image1.resize((new_width, new_height1))
    new_image2 = image2.resize((new_width, new_height2))
    
    # Convert images to PhotoImage format for tkinter display
    image1_tk = ImageTk.PhotoImage(new_image1)
    image2_tk = ImageTk.PhotoImage(new_image2)

    img_frame = LabelFrame(m)
    img_frame.pack(side='top')
    # Create labels for images
    label1 = Label(img_frame, image=image1_tk)
    label2 = Label(img_frame, image=image2_tk)

    # Pack the labels side-by-side (adjust padding as needed)
    label1.pack(side=LEFT, padx=10)
    label2.pack(side=LEFT, padx=10)

    cap_frame = LabelFrame(m)
    cap_frame.pack(side='top')
    caption_label1 = Label(cap_frame, text="Image with potential centroid points", width=60, wraplength=400)  # Adjust width and wraplength
    caption_label1.pack(side=LEFT,padx=10)
    caption_label2 = Label(cap_frame, text="Annotated image centroid points (red = rejected, green = accepted)", width=60, wraplength=400)  # Adjust width and wraplength
    caption_label2.pack(side=LEFT,padx=10)
    # Prevent image garbage collection
    label1.image = image1_tk  
    label2.image = image2_tk

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
