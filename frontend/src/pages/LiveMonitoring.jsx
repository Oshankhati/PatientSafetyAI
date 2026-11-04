import Icon from '../components/Icon';
import PoseCanvas from '../components/PoseCanvas';
import { Panel, PageHeading, MetaStrip, Pill, Eyebrow } from '../components/ui';
import { timeLabel } from '../utils';

export default function LiveMonitoring({ vm, th, updatedAt }) {
  const { pose } = vm;
  const score = pose?.score;
  const fall = pose?.fallConfirmed, high = !fall && pose?.level === 'High';
  return (
    <>
      <PageHeading title="Live Monitoring" sub="Real-time pose tracking and fall-risk monitoring" />
      <div className="focus-head">
        <div><Eyebrow>FOCUSED VIEW</Eyebrow><h2 className="h-md">{pose ? `${pose.room || '—'} · ${pose.patient || '—'}` : 'No active feed'}</h2></div>
        {score != null && <Pill level={pose.level}>{score}%</Pill>}
      </div>
      {fall && (
        <div className="banner fall-banner"><Icon name="alert" size={18} />
          <div><b>Fall detected</b><span>The model reports a confirmed fall for {pose.patient || 'the tracked patient'}. Check the patient immediately.</span></div></div>
      )}
      {high && (
        <div className="banner warn-banner"><Icon name="bell" size={18} />
          <div><b>High risk state detected</b><span>Monitor the patient closely. Risk engine score exceeds the configured {th.high.toFixed(2)} threshold.</span></div></div>
      )}
      <div className="live-grid">
        <Panel className="flush">
          <PoseCanvas keypoints={pose?.keypoints} height={340} aspect={pose?.aspect || 16 / 9} tag={pose?.keypoints ? `${(pose.patient || 'PERSON').toUpperCase()}${pose.detConf != null ? ` · ${pose.detConf}%` : ''}` : null} />
          <MetaStrip items={[[pose?.fps || '—', 'FPS'], [pose?.frame !== '' && pose?.frame != null ? pose.frame : '—', 'Frame'], [pose?.room || '—', 'Room'], [timeLabel(updatedAt), 'Timestamp']]} />
        </Panel>
        <div className="stack">
          <Panel><Eyebrow>AI PREDICTION</Eyebrow><h3 className="big">{pose?.prediction || '—'}</h3>
            <div className="kv"><span>Confidence</span><b>{pose?.predConf == null ? '—' : `${pose.predConf}%`}</b></div></Panel>
          <Panel><Eyebrow>RISK SCORE</Eyebrow><h3 className={`big ${pose?.level === 'High' ? 'warn' : ''}`}>{score == null ? '—' : `${score}%`}</h3>
            <div className="kv"><span>Risk level</span>{pose?.level ? <Pill level={pose.level} /> : <b>—</b>}</div></Panel>
        </div>
      </div>
    </>
  );
}
