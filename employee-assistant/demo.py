"""无需启动服务器：打印每次 graph 实际经过的节点。"""

from app.services.assistant import Assistant

if __name__ == "__main__":
    assistant = Assistant()
    cases = [
        ("上海员工去北京出差，住宿上限是多少？", "alice", "travel"),
        ("那深圳呢？", "alice", "travel"),
        ("年假怎么申请？", "alice", "leave"),
        ("年假怎么申请？", "alice", "leave"),
        ("年假怎么申请？", "bob", "leave"),
        ("薪酬审阅流程是什么？", "alice", "salary"),
        ("薪酬审阅流程是什么？", "helen", "salary"),
    ]
    for question, user, session in cases:
        result = assistant.ask(question, user, session)
        print(f"\n{'=' * 55}\n{user}: {question}")
        print("\n".join(result.trace))
        print(result.answer)
