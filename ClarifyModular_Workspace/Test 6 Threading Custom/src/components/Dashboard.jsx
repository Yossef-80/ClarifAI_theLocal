import React from "react";

const students = [
  { name: "Evelyn", status: "Not Attentive" },
  { name: "Isabelle", status: "Attentive" },
  { name: "John", status: "Not Attentive" },
  { name: "William", status: "Attentive" },
  { name: "Chloe", status: "Not Attentive" },
  { name: "Charlotte", status: "Attentive" },
  { name: "Caleb", status: "Not Attentive" },
  { name: "Olivia", status: "Not Attentive" },
  { name: "Mia", status: "Not Attentive" },
  { name: "Lucas", status: "Not Attentive" },
  { name: "Sophia", status: "Attentive" }
];

const alerts = [
  { time: "00:05:22", message: "Comprehension is high during experiments", type: "success" },
  { time: "00:11:35", message: "2 students are distracted", type: "warning" },
  { time: "00:19:10", message: "6 students are distracted", type: "danger" }
];

const transcript = [
  { time: "00:12:30", text: "When we add rough sandpaper, the car slows down!", score: 80 },
  { time: "00:14:15", text: "Friction is a force that works against motion.", score: 75 },
  { time: "00:16:40", text: "Can anyone think of examples of friction in everyday life?", score: 60 },
  { time: "00:19:10", text: "Great answers! Brakes on bikes use friction to stop.", score: 50 }
];

const Alert = ({ time, message, type }) => (
  <div className={`p-2 mb-1 rounded ${
    type === "success"
      ? "bg-green-100 text-green-800"
      : type === "warning"
      ? "bg-yellow-100 text-yellow-800"
      : "bg-red-100 text-red-800"
  }`}>
    [{time}] {message}
  </div>
);

const Dashboard = () => {
  return (
    <div className="grid grid-cols-3 gap-4 p-4 font-sans">
      {/* Left - Video Feed */}
      <div className="col-span-2">
        <div className="border rounded shadow">
          <div className="p-2 font-bold">Live Classroom Feed</div>
          <div className="grid grid-cols-4 gap-1 bg-black text-white text-xs">
            {students.map((student, i) => (
              <div
                key={i}
                className={`flex items-center justify-center h-20 border-2 m-1 rounded text-center ${
                  student.status === "Attentive"
                    ? "border-green-500"
                    : "border-red-500"
                }`}
              >
                {student.name} – {student.status}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right - Alerts and Metrics */}
      <div className="space-y-4">
        <div className="border rounded shadow p-2">
          <div className="font-bold mb-2">Alerts</div>
          {alerts.map((a, i) => (
            <Alert key={i} {...a} />
          ))}
        </div>

        <div className="border rounded shadow p-2">
          <div className="font-bold mb-2">Classroom Metrics</div>
          <div className="mb-2">
            <div className="text-sm mb-1">Attention Rate (50%)</div>
            <div className="w-full bg-gray-200 rounded h-3">
              <div className="h-3 bg-red-500 rounded w-[50%]" />
            </div>
          </div>
          <div className="mb-2">
            <div className="text-sm mb-1">Comprehension Rate (40%)</div>
            <div className="w-full bg-gray-200 rounded h-3">
              <div className="h-3 bg-red-500 rounded w-[40%]" />
            </div>
          </div>
          <div className="text-sm">Active Students 6 / 12</div>
        </div>
      </div>

      {/* Bottom - Transcript */}
      <div className="col-span-3 border rounded shadow p-2 mt-4">
        <div className="font-bold mb-2">Lesson Transcript</div>
        <div className="space-y-1">
          {transcript.map((item, i) => (
            <div
              key={i}
              className={`flex items-center justify-between p-2 rounded ${
                item.score >= 75
                  ? "bg-green-100"
                  : item.score >= 60
                  ? "bg-yellow-100"
                  : "bg-red-100"
              }`}
            >
              <div>
                [{item.time}] {item.text}
              </div>
              <div className="text-sm font-bold">{item.score}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
