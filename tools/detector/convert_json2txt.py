import json
import os
def video_id_process(video_path):
    mapping = {}

    id_csv = open(video_path, 'r')
    id_csv = id_csv.readlines()[1:]
    id_csv = [_sub.replace("\n", "") for _sub in id_csv]

    for text_id in id_csv:
        text_id = text_id.split(",")
        video_id = text_id[0]
        text_id = text_id[1:]
        for video_name in text_id:
            # video_name = video_name.lower().replace("noaudio_", "")
            video_name = video_name.replace(".mp4", "")
            mapping[video_name] = video_id

    return mapping


if __name__ == "__main__":
    submit_file = open("submit.txt", "a")
    submit_list = []

    video_id_mapping = video_id_process("tools/video_ids.csv")
    json_file = json.load(
        open("submit.json", "r"))

    for video_name, result in json_file['results'].items():
        video_id = video_id_mapping[video_name+".MP4"]

        for _result in result:  # check each bmn proposal

            start_time = int(_result["segment"][0])
            end_time = int(_result["segment"][1])

            # get only top-1
            class_id = str(int(_result['label']) + 1)
            line = video_id+" "+class_id+" " + \
                str(start_time)+" "+str(end_time)+"\n"

            submit_file.write(line)

    submit_file.close()