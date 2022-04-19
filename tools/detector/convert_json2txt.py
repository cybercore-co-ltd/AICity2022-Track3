import json
import argparse 

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

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--video_id', type=str, help='video_ids.csv',
                        default='tools/detector/video_ids_A2clean.csv')
    return parser.parse_args()

if __name__ == "__main__":
    submit_file = open("submit.txt", "a")
    submit_list = []
    args = parse_args()

    video_id_mapping = video_id_process(args.video_id)
    json_file = json.load(
        open("submit.json", "r"))
    set_lines = []
    for video_name, result in json_file['results'].items():
        video_id = video_id_mapping[video_name+".MP4"]

        for _result in result:  # check each bmn proposal

            start_time = int(_result["segment"][0])
            end_time = int(_result["segment"][1])

            # get only top-1
            class_id = str(int(_result['label']) + 1)
            line = video_id+" "+class_id+" " + \
                str(start_time)+" "+str(end_time)+"\n"
            if line in set_lines: continue
            set_lines.append(line)
            submit_file.write(line)

    submit_file.close()