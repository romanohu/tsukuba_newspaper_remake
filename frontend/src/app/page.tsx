// app/page.tsx
export default async function Page() {
  const q = "エヴァンゲリオン";
  const res = await fetch(`http://localhost:8000/search/?q=${encodeURIComponent(q)}&topk=5`);
  const data = await res.json();

  return (
    <div>
      <h1>検索結果</h1>
      <ul>
        {data.results.map((item: any, idx: number) => (
          <li key={idx}>{item.title || item.path}</li>
        ))}
      </ul>
    </div>
  );
}
