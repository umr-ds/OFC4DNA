with open("output.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("WARNING") or line.startswith("Could not"):
            continue
        elif "Decoding" in line:
            print(line.split("out\\")[1].split(" ")[0] + ", " + line.split("finished with overhead ")[1].split(" and")[0] + ", "
                  + line.split(" and unrecovered ")[1].split(" (")[0] + ", " + line.split(" (run ")[1].split(")")[0])
