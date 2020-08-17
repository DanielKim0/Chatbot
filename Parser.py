import json
import gzip
import os
import yaml


class Parser:
    """Class that handles parsing the data into a format readable by machine learning models."""

    def __init__(self, path):
        self.path = path

    def parse_simple(self):
        """Function that parses the 'simple' Chatterbot corpus. Corpus consists of several hundered question/answer
        pairs in .yaml file format separated into files stored in a directory."""
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
                    ans = ""
                    for rep in con[1:]:
                        ans += " " + rep
                    a = ans
                elif len(con) > 1:
                    a = con[1]

                if type(a) == str:
                    questions.append(q)
                    answers.append(a)

        return questions, answers

    def parse_google(self):
        """Function that parses Google's natural question corpus dataset, a more complex and larger dataset consisting
        of several hundred thousand Google searches as well as their corresponding answers found on wikipedia pages.
        Data is stored in one file in .json format, making it easily transportable to other files like it."""
        # line-by-line using file object as iterator to save memory
        questions = []
        answers = []

        lines_parsed = 0
        with open(self.path) as f:
            for line in f:
                que, ans = self.parse_line(line)
                questions.extend(que)
                answers.extend(ans)
                if lines_parsed % 1000 == 0:
                    print("Lines parsed: " + str(lines_parsed))
                lines_parsed += 1

        return questions, answers

    def parse_line(self, line):
        """Function that parses one line, or question/answer dataset, from the Google corpus file."""
        questions = []
        answers = []

        data = json.loads(line)
        question = data["question_text"]
        for item in data["long_answer_candidates"]:
            answer = " ".join(data["document_text"].split(" ")[int(item["start_token"]):int(item["end_token"])])
            questions.append(question)
            answers.append(answer)

        return questions, answers

    def store_data(self, questions, answers):
        """Function that stores question/answer data in two different gzipped files."""
        qfile = gzip.open("questions.gzip", "wt")
        qfile.write("\n".join(questions))

        afile = gzip.open("answers.gzip", "wt")
        afile.write("\n".join(answers))

    def main(self, google=False, store=False):
        if google:
            questions, answers = self.parse_google()
        else:
            questions, answers = self.parse_simple()
        if store:
            self.store_data(questions, answers)
        return questions, answers
