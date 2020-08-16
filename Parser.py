import json
import gzip
import os
import yaml

class Parser:
    def __init__(self, path):
        self.path = path

    def parse_simple(self):
        file_list = os.listdir(self.path)
        questions = []
        answers = []

        for path in file_list:
            stream = open(self.path + "/" + path, "rb")
            docs = yaml.safe_load(stream)
            conversations = docs['conversations']
            for con in conversations:
                q = con[0]
                if len(con) > 2:
                    replies = con[1:]
                    ans = ""
                    for rep in replies:
                        ans += " " + rep
                    a = ans
                elif len(con) > 1:
                    a = con[1]

                if a and type(a) == str:
                    questions.append(q)
                    answers.append(a)

        return questions, answers

    def parse_google(self, path):
        # line-by-line using file object as iterator to save memory
        questions = []
        answers = []

        lines_parsed = 0
        with open("log.txt") as f:
            for line in f:
                que, ans = parse_line(line)
                questions.extend(que)
                answers.extend(ans)
                if lines_parsed % 1000 == 0:
                    print("Lines parsed: " + str(lines_parsed) + "\n")

        return questions, answers

    def parse_line(self, line):
        questions = []
        answers = []

        data = json.loads(line)
        question = data["question_text"]
        for item in data["long answer candidates"]:
            answer = " ".join(data["document_text"].split(" ")[int(item["start_token"]):int(item["end_token"])])
            questions.append(question)
            answers.append(answer)

        return questions, answers

    def store_data(self, questions, answers):
        qfile = gzip.open("questions.gzip", "wt")
        qfile.write("\n".join(questions))

        afile = gzip.open("answers.gzip", "wt")
        afile.write("\n".join(answers))

    def main(self):
        questions, answers = self.parse_simple()
        self.store_data(questions, answers)
        return questions, answers
