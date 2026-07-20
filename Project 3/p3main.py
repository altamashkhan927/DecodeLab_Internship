import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_user_skills():
    print("Enter your skills one by one (minimum 3).")
    print("Type 'done' when finished.\n")

    skills = []

    while True:
        skill = input(f"Skill #{len(skills)+1}: ").strip().lower()
        if skill == "done":
            if len(skills) < 3:
                print("Please enter at least 3 skills.")
                continue
            break
        if skill and skill not in skills:
            skills.append(skill)

    return skills


def load_dataset(path="raw_skills.csv"):
    try:
        data = pd.read_csv(path)
        return data

    except FileNotFoundError:
        print("Dataset not found.")
        exit()


def recommend_roles(data, user_skills, top_n=3):
    role_documents = data["skills"].str.lower().str.replace(",", " ")

    user_document = " ".join(user_skills)

    corpus = list(role_documents) + [user_document]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(corpus)

    role_vectors = tfidf_matrix[:-1]

    user_vector = tfidf_matrix[-1]

    scores = cosine_similarity(user_vector, role_vectors).flatten()

    data["Similarity"] = scores
    if scores.max() == 0:
        return data.head(top_n), True

    recommendations = data.sort_values(
        by="Similarity",
        ascending=False
    )
    return recommendations.head(top_n), False


def main():
    dataset = load_dataset()
    user_skills = get_user_skills()

    print("\nYour Skills:")
    print(", ".join(user_skills))
    print("\nFinding the best career paths...\n")

    recommendations, fallback = recommend_roles(
        dataset,
        user_skills
    )

    if fallback:
        print("No matching skills found.")
        print("Showing default career paths:\n")
    else:
        print("Top Recommended Career Paths:\n")

    for i, row in enumerate(recommendations.itertuples(), start=1):

        print(f"{i}. {row.job_role}")
        print(f"   Match: {row.Similarity*100:.1f}%")
        print(f"   Skills: {row.skills}")
        print()


if __name__ == "__main__":
    main()
